package neuralnet

import (
	"testing"
	"math"
)

// TestSigmoid tests the sigmoid function
func TestSigmoid(t *testing.T) {
	tests := []struct {
		input    float64
		expected float64
	}{
		{0, 0.5}, // Sigmoid(0) should return 0.5
		{1, 0.7310585786300049}, // Sigmoid(1) should return approximately 0.731
		{-1, 0.2689414213699951}, // Sigmoid(-1) should return approximately 0.269
	}

	for _, test := range tests {
		t.Run("Test Sigmoid", func(t *testing.T) {
			got := sigmoid(test.input)
			if math.Abs(got-test.expected) > 0.00001 {
				t.Errorf("For input %f, expected %f but got %f", test.input, test.expected, got)
			}
		})
	}
}

// TestSigmoidDerivative tests the sigmoidDerivative function
func TestSigmoidDerivative(t *testing.T) {
	tests := []struct {
		input    float64
		expected float64
	}{
		{0, 0.
