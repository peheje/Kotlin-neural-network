package Neural

class Layer {
    val neurons: Array<Neuron>
    private var size: Int

    constructor(previousInputSize: Int, size: Int) {
        this.size = size
        this.neurons = Array(size) { Neuron(previousInputSize) }
    }

    constructor(neurons: Array<Neuron>) {
        this.size = neurons.size
        this.neurons = neurons
    }

    fun copy(): Layer {
        return Layer(Array(this.size) { i -> neurons[i].copy() })
    }

    operator fun invoke(inputs: DoubleArray): DoubleArray {
        return DoubleArray(size) { i -> neurons[i](inputs) }
    }
}