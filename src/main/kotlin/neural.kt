import java.util.concurrent.ThreadLocalRandom

fun random(from: Double, to: Double): Double {
    return ThreadLocalRandom.current().nextDouble(from, to)
}

class Neuron {
    private var weights: DoubleArray
    private var bias: Double

    constructor(numWeights: Int) {
        this.weights = DoubleArray(numWeights) { random(-1.0, 1.0) }
        this.bias = random(-1.0, 1.0)
    }

    private constructor(weight: DoubleArray, bias: Double) {
        this.weights = weight
        this.bias = bias
    }

    fun copy(): Neuron {
        return Neuron(weights.copyOf(), bias)
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
}

class Layer {
    private val neurons: Array<Neuron>
    private var size: Int

    constructor(previousInputSize: Int, size: Int) {
        this.size = size
        this.neurons = Array(size) { Neuron(previousInputSize) }
    }

    private constructor(neurons: Array<Neuron>) {
        this.size = neurons.size
        this.neurons = neurons
    }

    fun copy(): Layer {
        val neuronsCopy = Array(this.size) { i -> neurons[i].copy() }
        return Layer(neuronsCopy)
    }

    operator fun invoke(inputs: DoubleArray): DoubleArray {
        return DoubleArray(size) { i -> neurons[i](inputs) }
    }
}

class Net {
    private val l1 = Layer(2, 4)
    private val l2 = Layer(4, 4)
    private val l3 = Layer(4, 2)
    private val layers: Array<Layer>

    constructor() {
        layers = arrayOf(l1, l2, l3)
    }

    private operator fun invoke(inputs: DoubleArray): DoubleArray {
        val o1 = l1(inputs)
        val o2 = l2(o1)
        val o3 = l3(o2)
        return o3
    }

    private fun softmax(netOutput: DoubleArray): DoubleArray {
        val max = netOutput.max() ?: 0.0
        for (i in 0 until netOutput.size) netOutput[i] -= max
        val sum = netOutput.sumByDouble { Math.exp(it) }
        return DoubleArray(netOutput.size) { i -> Math.exp(netOutput[i]) / sum }
    }

    fun softmaxLoss(inputs: DoubleArray, correctIndex: Int): Double {
        val netOutput: DoubleArray = this(inputs)
        val sm = softmax(netOutput)[correctIndex]
        return -Math.log(sm)
    }
}
