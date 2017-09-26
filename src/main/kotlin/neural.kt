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
        return Math.tanh(sum)
        //return 1.0 / (1.0 + Math.exp(-sum))
    }

    fun mutate(mutateRate: Double, mutateFreq: Double) {
        for (i in 0 until weights.size) if (random() < mutateFreq)
            weights[i] += random(-mutateRate, mutateRate)
        bias += random(-mutateRate, mutateRate)
    }

    fun crossover(net: List<Net>, layerIdx: Int, neuronIdx: Int, crossoverRate: Double, crossoverFreq: Double) {
        val mate: Neuron = Net.pick(net).layers[layerIdx].neurons[neuronIdx]
        for (i in 0 until weights.size) if (random() < crossoverFreq)
            weights[i] = Net.lerp(weights[i], mate.weights[i], random(0.0, crossoverRate))
        bias = Net.lerp(mate.bias, bias, random(0.0, crossoverRate))
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
        return Layer(Array(this.size) { i -> neurons[i].copy() })
    }

    operator fun invoke(inputs: DoubleArray): DoubleArray {
        return DoubleArray(size) { i -> neurons[i](inputs) }
    }
}

class Net {

    val layers: Array<Layer>

    constructor(layers: Array<Layer>) {
        this.layers = layers
    }

    constructor() {
        // Standard size
        layers = arrayOf(
                Layer(2, 4),
                Layer(4, 8),
                Layer(8, 4),
                Layer(4, 2))

    }

    var fitness: Double = 0.0

    companion object {
        private var wheel = DoubleArray(0)

        fun computeWheel(arr: List<Net>) {
            var sum = 0.0
            wheel = DoubleArray(arr.size) { i -> sum += arr[i].fitness; sum }
        }

        fun pick(arr: List<Net>): Net {
            val sum = wheel.last()
            val r = random() * sum
            var idx = wheel.binarySearch(r)
            if (idx < 0) idx = -idx - 1
            return arr[idx].copy()
        }

        fun lerp(a: Double, b: Double, p: Double): Double {
            return a + (b - a) * p
        }
    }



    private fun copy(): Net {
        return Net(Array(layers.size) { i -> layers[i].copy() })
    }

    fun computeFitness(input: DoubleArray, target: DoubleArray) {
        fitness = 1.0 / (error(input, target) + 1.0)
    }

    fun error(input: DoubleArray, target: DoubleArray): Double {
        val netOutput = this(input)
        return 0.5 * (0 until target.size).sumByDouble { Math.pow(target[it] - netOutput[it], 2.0) }
    }

    operator fun invoke(inputs: DoubleArray): DoubleArray {
        var layerOutput: DoubleArray = inputs
        for (layer in layers) layerOutput = layer(layerOutput)
        return layerOutput
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

    override fun toString(): String {
        val sb = StringBuilder()
        for (layer in layers) {
            for (neuron in layer.neurons) {
                sb.append(neuron.toString())
            }
        }
        return sb.toString()
    }

    fun crossover(pool: List<Net>, crossoverFreq: Double, crossoverRate: Double) {
        for ((layerIdx, layer) in layers.withIndex()) {
            for ((neuronIdx, neuron) in layer.neurons.withIndex()) {
                neuron.crossover(pool, layerIdx, neuronIdx, crossoverRate, crossoverFreq)
            }
        }
    }

    fun mutate(mutateFreq: Double, mutateRate: Double) {
        for ((layerIdx, layer) in layers.withIndex()) {
            for ((neuronIdx, neuron) in layer.neurons.withIndex()) {
                neuron.mutate(mutateRate, mutateFreq)
            }
        }
    }
}