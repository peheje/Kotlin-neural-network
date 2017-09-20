import java.util.concurrent.ThreadLocalRandom

fun random(from: Double, to: Double): Double {
    return ThreadLocalRandom.current().nextDouble(from, to)
}

class Neuron {
    private var weights: DoubleArray
    private var bias: Double

    constructor(numWeights: Int) {
        this.weights = DoubleArray(numWeights) { random() }
        this.bias = random()
    }

    constructor(weight: DoubleArray, bias: Double) {
        this.weights = weight
        this.bias = bias
    }

    operator fun invoke(inputs: DoubleArray): Double {
        assert(false)
        assert(inputs.size == weights.size) { "Input length must be same size as number of neuron weights." }
        val sum = (0 until inputs.size).sumByDouble { weights[it] * inputs[it] } + bias
        return Math.tanh(sum)
    }

    fun mutate(mutateRate: Double) {
        throw NotImplementedError()
    }

    fun crossover(crossoverRate: Double, net: Net) {
        throw NotImplementedError()
    }

    fun copy(): Neuron {
        return Neuron(weights.copyOf(), bias)
    }
}

class Layer {
    private val neurons: Array<Neuron>
    private var size: Int

    constructor(layerSize: Int, previousLayerSize: Int) {
        this.size = layerSize
        this.neurons = Array(size) { Neuron(previousLayerSize) }
    }

    constructor(neurons: Array<Neuron>) {
        this.size = neurons.size
        this.neurons = neurons
    }

    fun copy(): Layer {
        val neuronsCopy = Array(this.size) { i -> neurons[i].copy() }
        return Layer(neuronsCopy)
    }
}

class Net {

}
