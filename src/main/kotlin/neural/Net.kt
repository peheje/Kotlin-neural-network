package neural

import random
import java.util.concurrent.ThreadLocalRandom

class Net {
    val layers: Array<Layer>
    var fitness: Double = 0.0

    constructor(layers: Array<Layer>, fitness: Double) {
        this.layers = layers
        this.fitness = fitness
    }

    constructor(trainingXs: Array<DoubleArray>, trainingYs: Array<DoubleArray>, layerSetup: List<Int>, parentInheritance: Double) {
        layers = (0 until layerSetup.size - 1).map { Layer(layerSetup[it], layerSetup[it + 1]) }.toTypedArray()
        computeFitness(trainingXs, trainingYs, parentInheritance)
    }

    private fun copy(): Net {
        return Net(Array(layers.size) { i -> layers[i].copy() }, fitness)
    }

    fun computeFitness(trainingXs: Array<DoubleArray>, trainingYs: Array<DoubleArray>, parentInheritance: Double, batchSize: Int = 0) {

        var trainingXs = trainingXs
        var trainingYs = trainingYs
        val batchSize = Math.min(batchSize, trainingXs.size)

        if (batchSize != 0) {
            val batchXs = mutableListOf<DoubleArray>()
            val batchYs = mutableListOf<DoubleArray>()
            while (batchXs.size < batchSize) {
                val r = ThreadLocalRandom.current().nextInt(trainingXs.size)
                batchXs.add(trainingXs[r])
                batchYs.add(trainingYs[r])
            }
            trainingXs = batchXs.toTypedArray()
            trainingYs = batchYs.toTypedArray()
        }

        val sumErr = (0 until trainingXs.size).sumByDouble { error(trainingXs[it], trainingYs[it]) }
        val batchFitness = 1.0 / (sumErr + 1.0)
        val parentFitness = fitness
        fitness = parentFitness * parentInheritance + batchFitness
    }

    private fun error(input: DoubleArray, target: DoubleArray): Double {
        val netOutput = this(input)
        return (0 until target.size).sumByDouble { Math.pow(target[it] - netOutput[it], 2.0) }
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

    fun crossover(pool: List<Net>, crossoverRate: Double) {
        for ((layerIdx, layer) in layers.withIndex()) {
            for ((neuronIdx, neuron) in layer.neurons.withIndex()) {
                neuron.crossover(pool, layerIdx, neuronIdx, crossoverRate)
            }
        }
    }

    fun mutate(mutateFreq: Double, mutateRate: Double) {
        for (layer in layers) {
            for (neuron in layer.neurons) {
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
    }
}