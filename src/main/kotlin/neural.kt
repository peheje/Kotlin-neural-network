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

    constructor(weights: DoubleArray, bias: Double) {
        this.weights = weights
        this.bias = bias
    }

    fun copy(): Neuron {
        return Neuron(weights.copyOf(), bias)
    }

    operator fun invoke(inputs: DoubleArray): Double {
        val sum = (0 until inputs.size).sumByDouble { weights[it] * inputs[it] } + bias
        val logistic = (1.0/(1.0+Math.exp(-sum)))
        // println(" logistic(${inputs.toList()} * ${weights.toList()} + $bias) = $logistic")
        return logistic
    }

    fun mutate(mutateRate: Double) {
        for (i in 0 until weights.size) weights[i] += random(-mutateRate, mutateRate)
        bias += random(-mutateRate, mutateRate)
    }

    fun crossover(crossoverRate: Double, net: Net) {
        throw NotImplementedError()
    }

    override fun toString(): String {
        val sb = StringBuilder()
        sb.append("[")
        for ((i, w) in weights.withIndex()) {
            sb.append(w.toString())
            if (i < weights.size - 1) sb.append(", ")
        }
        sb.append("]")
        return sb.toString()
    }
}

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
        val neuronsCopy = Array(this.size) { i -> neurons[i].copy() }
        return Layer(neuronsCopy)
    }

    operator fun invoke(inputs: DoubleArray): DoubleArray {
        return DoubleArray(size) { i -> neurons[i](inputs) }
    }
}

class Net(private val layers: Array<Layer>) {

    operator fun invoke(inputs: DoubleArray): DoubleArray {
        var layerOutput: DoubleArray = inputs
        for (layer in layers) layerOutput = layer(layerOutput)
        return layerOutput
    }

    fun error(input: DoubleArray, target: DoubleArray): Double {
        val netOutput = this(input)
        return 0.5 * (0 until target.size).sumByDouble { Math.pow(target[it] - netOutput[it], 2.0) }
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

    fun mutateAll(mutateRate: Double) {
        for (layer in layers) {
            for (neuron in layer.neurons) {
                neuron.mutate(mutateRate)
            }
        }
    }

    override fun toString(): String {
        val sb = StringBuilder()
        for (layer in layers) {
            for (neuron in layer.neurons) {
                sb.append(neuron.toString())
            }
        }
        return sb.toString()
    }
}