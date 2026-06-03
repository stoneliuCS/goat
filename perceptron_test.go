package goat_test

import (
	"testing"

	nn "goat"

	"github.com/stretchr/testify/assert"
)

// We can model the perceptron as a NAND Gate
func TestNANDGate(t *testing.T) {
	var bias float64 = 3
	p1 := nn.CreatePerceptron(bias, 2)

	// NAND GATE

	// 0 NAND 0 => 1
	assert.Equal(t, uint8(1), p1.Forward([]nn.Input{
		{X: 0, W: -2},
		{X: 0, W: -2},
	}))

	// 0 NAND 1 => 1
	assert.Equal(t, uint8(1), p1.Forward([]nn.Input{
		{X: 0, W: -2},
		{X: 1, W: -2},
	}))

	// 1 NAND 1 => 0
	assert.Equal(t, uint8(0), p1.Forward([]nn.Input{
		{X: 1, W: -2},
		{X: 1, W: -2},
	}))

	// 0 NAND 0 => 1
	assert.Equal(t, uint8(1), p1.Forward([]nn.Input{
		{X: 0, W: -2},
		{X: 0, W: -2},
	}))
}

func TestXOR(t *testing.T) {

	// Building the network
	var bias float64 = 3
	p1 := nn.CreatePerceptron(bias, 2)
	p2 := nn.CreatePerceptron(bias, 2)
	p3 := nn.CreatePerceptron(bias, 2)
	p4 := nn.CreatePerceptron(bias, 2)
	p5 := nn.CreatePerceptron(bias, 2)

}
