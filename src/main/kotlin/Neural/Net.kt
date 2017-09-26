package Neural

import random

class Net {
    val layers: Array<Layer>
    var fitness: Double = 0.0

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
}