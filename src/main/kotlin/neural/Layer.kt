package neural

class Layer {
    val neurons: Array<Neuron>
    private val isLast: Boolean

    constructor(previousInputSize: Int, size: Int, lastLayer: Boolean) {
        neurons = Array(size) { Neuron(previousInputSize) }
        isLast = lastLayer
    }

    private constructor(otherNeurons: Array<Neuron>, lastLayer: Boolean) {
        neurons = otherNeurons
        isLast = lastLayer
    }

    fun copy(): Layer {
        return Layer(Array(neurons.size) { i -> neurons[i].copy() }, isLast)
    }

    operator fun invoke(inputs: DoubleArray): DoubleArray {
        return DoubleArray(neurons.size) { i -> neurons[i](inputs, isLast) }
    }
}