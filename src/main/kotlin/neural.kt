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

    constructor(previousInputSize: Int, layerSize: Int) {
        this.size = layerSize
        this.neurons = Array(size) { Neuron(previousInputSize) }
    }

    private constructor(neurons: Array<Neuron>) {
        this.size = neurons.size
        this.neurons = neurons
    }

    operator fun invoke(inputs: DoubleArray): DoubleArray {
        return DoubleArray(size) { i -> neurons[i](inputs) }
    }

    fun copy(): Layer {
        val neuronsCopy = Array(this.size) { i -> neurons[i].copy() }
        return Layer(neuronsCopy)
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

    private fun forwardPass(inputs: DoubleArray): DoubleArray {
        val o1 = l1(inputs)
        val o2 = l2(o1)
        val o3 = l3(o2)
        return o3
    }

    private fun softmax(netBelief: DoubleArray): DoubleArray {
        val max = netBelief.max() ?: 0.0
        for (i in 0 until netBelief.size) netBelief[i] -= max
        val sum = netBelief.sumByDouble { Math.exp(it) }
        return DoubleArray(netBelief.size) { i -> Math.exp(netBelief[i]) / sum }
    }

    fun softmaxLoss(inputs: DoubleArray, correctIndex: Int): Double {
        val netBelief: DoubleArray = forwardPass(inputs)
        val sm = softmax(netBelief)[correctIndex]
        return -Math.log(sm)
    }
}
