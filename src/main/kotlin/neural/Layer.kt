package neural

class Layer {
    val neurons: Array<Neuron>

    constructor(previousInputSize: Int, size: Int) {
        neurons = Array(size) { Neuron(previousInputSize) }
    }

    private constructor(otherNeurons: Array<Neuron>) {
        neurons = otherNeurons
    }

    fun copy(): Layer {
        return Layer(Array(neurons.size) { i -> neurons[i].copy() })
    }

    operator fun invoke(inputs: DoubleArray): DoubleArray {
        return DoubleArray(neurons.size) { i -> neurons[i](inputs) }
    }
}