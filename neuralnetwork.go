package neuralnet

import (
	"math/rand"
	"time"
	"math"
)

// Sigmoid activation function
func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// Sigmoid derivative function for backpropagation
func sigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}

// NeuralNet struct representing the neural network
type NeuralNet struct {
	inputLayer  int
	hiddenLayer int
	outputLayer int
	weightsIH   [][]float64 // Weights from input to hidden
	weightsHO   [][]float64 // Weights from hidden to output
	biasH       []float64   // Biases for the hidden layer
	biasO       []float64   // Biases for the output layer
	learningRate float64
}

// NewNeuralNet creates a new neural network instance
func NewNeuralNet(inputLayer, hiddenLayer, outputLayer int, learningRate float64) *NeuralNet {
	nn := &NeuralNet{
		inputLayer:  inputLayer,
		hiddenLayer: hiddenLayer,
		outputLayer: outputLayer,
		learningRate: learningRate,
	}

	// Initialize weights and biases with random values
	nn.weightsIH = make([][]float64, inputLayer)
	for i := range nn.weightsIH {
		nn.weightsIH[i] = make([]float64, hiddenLayer)
		for j := range nn.weightsIH[i] {
			nn.weightsIH[i][j] = rand.Float64()*2 - 1
		}
	}

	nn.weightsHO = make([][]float64, hiddenLayer)
	for i := range nn.weightsHO {
		nn.weightsHO[i] = make([]float64, outputLayer)
		for j := range nn.weightsHO[i] {
			nn.weightsHO[i][j] = rand.Float64()*2 - 1
		}
	}

	nn.biasH = make([]float64, hiddenLayer)
	nn.biasO = make([]float64, outputLayer)

	rand.Seed(time.Now().UnixNano())
	return nn
}

// Feedforward function that calculates the network output
func (nn *NeuralNet) FeedForward(input []float64) []float64 {
	// Input to hidden layer
	hidden := make([]float64, nn.hiddenLayer)
	for i := 0; i < nn.hiddenLayer; i++ {
		sum := 0.0
		for j := 0; j < nn.inputLayer; j++ {
			sum += input[j] * nn.weightsIH[j][i]
		}
		hidden[i] = sigmoid(sum + nn.biasH[i])
	}

	// Hidden to output layer
	output := make([]float64, nn.outputLayer)
	for i := 0; i < nn.outputLayer; i++ {
		sum := 0.0
		for j := 0; j < nn.hiddenLayer; j++ {
			sum += hidden[j] * nn.weightsHO[j][i]
		}
		output[i] = sigmoid(sum + nn.biasO[i])
	}

	return output
}

// Backpropagation function to adjust weights and biases
func (nn *NeuralNet) Backpropagate(input, target []float64) {
	// Feedforward
	hidden := make([]float64, nn.hiddenLayer)
	for i := 0; i < nn.hiddenLayer; i++ {
		sum := 0.0
		for j := 0; j < nn.inputLayer; j++ {
			sum += input[j] * nn.weightsIH[j][i]
		}
		hidden[i] = sigmoid(sum + nn.biasH[i])
	}

	output := make([]float64, nn.outputLayer)
	for i := 0; i < nn.outputLayer; i++ {
		sum := 0.0
		for j := 0; j < nn.hiddenLayer; j++ {
			sum += hidden[j] * nn.weightsHO[j][i]
		}
		output[i] = sigmoid(sum + nn.biasO[i])
	}

	// Output layer error and gradient
	outputError := make([]float64, nn.outputLayer)
	for i := 0; i < nn.outputLayer; i++ {
		outputError[i] = target[i] - output[i]
	}

	outputGradient := make([]float64, nn.outputLayer)
	for i := 0; i < nn.outputLayer; i++ {
		outputGradient[i] = outputError[i] * sigmoidDerivative(output[i])
	}

	// Hidden layer error and gradient
	hiddenError := make([]float64, nn.hiddenLayer)
	for i := 0; i < nn.hiddenLayer; i++ {
		sum := 0.0
		for j := 0; j < nn.outputLayer; j++ {
			sum += outputGradient[j] * nn.weightsHO[i][j]
		}
		hiddenError[i] = sum
	}

	hiddenGradient := make([]float64, nn.hiddenLayer)
	for i := 0; i < nn.hiddenLayer; i++ {
		hiddenGradient[i] = hiddenError[i] * sigmoidDerivative(hidden[i])
	}

	// Update weights and biases (using gradient descent)
	for i := 0; i < nn.hiddenLayer; i++ {
		for j := 0; j < nn.outputLayer; j++ {
			nn.weightsHO[i][j] += nn.learningRate * outputGradient[j] * hidden[i]
		}
		nn.biasO[i] += nn.learningRate * outputGradient[i]
	}

	for i := 0; i < nn.inputLayer; i++ {
		for j := 0; j < nn.hiddenLayer; j++ {
			nn.weightsIH[i][j] += nn.learningRate * hiddenGradient[j] * input[i]
		}
	}
	for i := 0; i < nn.hiddenLayer; i++ {
		nn.biasH[i] += nn.learningRate * hiddenGradient[i]
	}
}

// Train the network for a certain number of iterations
func (nn *NeuralNet) Train(inputs [][]float64, targets [][]float64, iterations int) {
	for i := 0; i < iterations; i++ {
		for j := 0; j < len(inputs); j++ {
			nn.Backpropagate(inputs[j], targets[j])
		}
	}
}
