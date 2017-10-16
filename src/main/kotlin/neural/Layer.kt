package neural

class Layer {
    val neurons: MutableList<Neuron>
    private val isLast: Boolean

    constructor(previousInputSize: Int, size: Int, lastLayer: Boolean) {
        neurons = MutableList(size) { Neuron(previousInputSize) }
        isLast = lastLayer
    }

    private constructor(otherNeurons: MutableList<Neuron>, lastLayer: Boolean) {
        neurons = otherNeurons
        isLast = lastLayer
    }

    fun copy(): Layer {
        return Layer(MutableList(neurons.size) { i -> neurons[i].copy() }, isLast)
    }

    operator fun invoke(inputs: DoubleArray): DoubleArray {
        return DoubleArray(neurons.size) { i -> neurons[i](inputs, isLast) }
    }
}