package goat_test

import (
	"testing"

	nn "goat"

	"github.com/stretchr/testify/assert"
)

// We can model the perceptron as a NAND Gate
func TestNANDGate(t *testing.T) {
	// 1 NAND 0 => 1
	assert.Equal(t, uint8(1), nn.NAND(1, 0))

	// 0 NAND 1 => 1
	assert.Equal(t, uint8(1), nn.NAND(0, 1))

	// 1 NAND 1 => 0
	assert.Equal(t, uint8(0), nn.NAND(1, 1))

	// 0 NAND 0 => 1
	assert.Equal(t, uint8(1), nn.NAND(0, 0))
}

func TestXOR(t *testing.T) {
	// 0 XOR 0 => 0
	assert.Equal(t, uint8(0), nn.XOR(0, 0))

	// 1 XOR 0 => 1
	assert.Equal(t, uint8(1), nn.XOR(1, 0))

	// 0 XOR 1 => 1
	assert.Equal(t, uint8(1), nn.XOR(0, 1))

	// 1 XOR 1 => 0
	assert.Equal(t, uint8(0), nn.XOR(1, 1))
}
