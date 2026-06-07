package goat

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"strings"
)

type Sample[N Number] struct {
	Image *Vector[N]
	Label *Vector[N]
	Digit uint8
}

func LoadMNIST[N Number](imagePath, labelPath string) ([]Sample[N], error) {
	images, rows, cols, err := readImages[N](imagePath)
	if err != nil {
		return nil, err
	}
	labels, err := readLabels(labelPath)
	if err != nil {
		return nil, err
	}
	if len(images) != len(labels) {
		return nil, fmt.Errorf("image/label count mismatch: %d vs %d", len(images), len(labels))
	}

	samples := make([]Sample[N], len(images))
	for i := range images {
		pixels := make([]N, rows*cols)
		for j, p := range images[i] {
			pixels[j] = N(p) / 255
		}
		oneHot := make([]N, 10)
		oneHot[labels[i]] = 1
		samples[i] = Sample[N]{
			Image: CreateMatrix([][]N{pixels}),
			Label: CreateMatrix([][]N{oneHot}),
			Digit: labels[i],
		}
	}
	return samples, nil
}

func openMaybeGzip(path string) (io.ReadCloser, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	if !strings.HasSuffix(path, ".gz") {
		return f, nil
	}
	gz, err := gzip.NewReader(f)
	if err != nil {
		f.Close()
		return nil, err
	}
	// Closing gz does not close f, so wrap both.
	return &gzipReadCloser{gz: gz, f: f}, nil
}

type gzipReadCloser struct {
	gz *gzip.Reader
	f  *os.File
}

func (r *gzipReadCloser) Read(p []byte) (int, error) { return r.gz.Read(p) }
func (r *gzipReadCloser) Close() error {
	err := r.gz.Close()
	if ferr := r.f.Close(); err == nil {
		err = ferr
	}
	return err
}

func readImages[N Number](path string) ([][]byte, int, int, error) {
	r, err := openMaybeGzip(path)
	if err != nil {
		return nil, 0, 0, err
	}
	defer r.Close()

	var header struct {
		Magic uint32
		Count uint32
		Rows  uint32
		Cols  uint32
	}
	if err := binary.Read(r, binary.BigEndian, &header); err != nil {
		return nil, 0, 0, err
	}
	if header.Magic != 0x00000803 {
		return nil, 0, 0, fmt.Errorf("bad image magic 0x%08x", header.Magic)
	}

	rows, cols := int(header.Rows), int(header.Cols)
	images := make([][]byte, header.Count)
	for i := range images {
		buf := make([]byte, rows*cols)
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, 0, 0, fmt.Errorf("reading image %d: %w", i, err)
		}
		images[i] = buf
	}
	return images, rows, cols, nil
}

func readLabels(path string) ([]uint8, error) {
	r, err := openMaybeGzip(path)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	var header struct {
		Magic uint32
		Count uint32
	}
	if err := binary.Read(r, binary.BigEndian, &header); err != nil {
		return nil, err
	}
	if header.Magic != 0x00000801 {
		return nil, fmt.Errorf("bad label magic 0x%08x", header.Magic)
	}

	labels := make([]uint8, header.Count)
	if _, err := io.ReadFull(r, labels); err != nil {
		return nil, err
	}
	return labels, nil
}
