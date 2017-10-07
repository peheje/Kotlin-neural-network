package neural

import org.nd4j.linalg.api.ops.impl.transforms.SoftMax
import random
import java.util.concurrent.ThreadLocalRandom

class Net {
    val layers: List<Layer>
    var fitness: Double = 0.0

    constructor(trainingXs: List<DoubleArray>, trainingYs: List<Int>, layerSetup: List<Int>, parentInheritance: Double, gamma: Double) {
        val lastLayerIdx = layerSetup.size - 2  // -2 as last is output size
        layers = (0 until layerSetup.size - 1).map { Layer(layerSetup[it], layerSetup[it + 1], it == lastLayerIdx) }
        computeFitness(trainingXs, trainingYs, parentInheritance, gamma)
    }

    private constructor(layers: List<Layer>, fitness: Double) {
        this.layers = layers
        this.fitness = fitness
    }

    private fun copy(): Net {
        return Net(List(layers.size) { i -> layers[i].copy() }, fitness)
    }

    operator fun invoke(inputs: DoubleArray): DoubleArray {
        var layerOutput = inputs
        for (layer in layers) layerOutput = layer(layerOutput)
        return layerOutput
    }

    fun computeFitness(xs: List<DoubleArray>, ys: List<Int>, parentInheritance: Double, gamma: Double) {
        // Todo move to crossoverAndMutate?

        var regularizationLoss = 0.0
        if (gamma != 0.0) {
            for (layer in layers)
                for (neuron in layer.neurons)
                    for (weight in neuron.weights)
                        regularizationLoss += weight * weight
        }

        regularizationLoss *= gamma

        val dataLoss = (0 until xs.size).sumByDouble { softmaxLoss(xs[it], ys[it]) } / xs.size
        val correctFitness = (nCorrectPredictions(xs, ys) / xs.size)

        var batchfitness = correctFitness - regularizationLoss - dataLoss
        batchfitness = Math.max(0.0, batchfitness)
        val parentFitness = fitness
        fitness = parentFitness * parentInheritance + batchfitness
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
        val mate = pick(pool)
        for ((layerIdx, layer) in layers.withIndex()) {
            for ((neuronIdx, neuron) in layer.neurons.withIndex()) {
                neuron.crossover(mate, layerIdx, neuronIdx, crossoverRate)
            }
        }
    }

    fun mutate(mutateFreq: Double, mutatePower: Double) {
        for (layer in layers) {
            for (neuron in layer.neurons) {
                neuron.mutate(mutatePower, mutateFreq)
            }
        }
    }

    companion object {
        private var wheel = DoubleArray(0)

        fun computeWheel(pool: List<Net>) {
            var sum = 0.0
            wheel = DoubleArray(pool.size) { i -> sum += pool[i].fitness; sum }
        }

        fun pick(arr: List<Net>): Net {
            val sum = wheel.last()
            val r = random() * sum
            var idx = wheel.binarySearch(r)
            if (idx < 0) idx = -idx - 1
            return arr[idx].copy()
        }

        fun crossoverAndMutate(net: Net, pool: List<Net>, crossoverProp: Double, crossoverRate: Double, mutateProp: Double, mutateFreq: Double, mutatePower: Double) {
            val mate = pick(pool)
            for ((layerIdx, layer) in net.layers.withIndex()) {
                for ((neuronIdx, neuron) in layer.neurons.withIndex()) {
                    if (random() < crossoverProp) neuron.crossover(mate, layerIdx, neuronIdx, crossoverRate)
                    if (random() < mutateProp) neuron.mutate(mutatePower, mutateFreq)
                }
            }
        }

        fun createBatch(xs: List<DoubleArray>,
                        ys: List<Int>,
                        batchSize: Int): Pair<List<DoubleArray>, List<Int>> {

            val clippedBatchSize = Math.min(batchSize, xs.size)

            val batchXs = mutableListOf<DoubleArray>()
            val batchYs = mutableListOf<Int>()
            while (batchXs.size < clippedBatchSize) {
                // Todo diversity in batch
                val r = ThreadLocalRandom.current().nextInt(xs.size)
                batchXs.add(xs[r])
                batchYs.add(ys[r])
            }
            return Pair(batchXs, batchYs)
        }
    }

    private fun nCorrectPredictions(xs: List<DoubleArray>, ys: List<Int>): Int {
        var nCorrect = 0
        for ((i, x) in xs.withIndex()) {
            val correct = ys[i]
            val neuralGuesses: DoubleArray = this(x)
            val bestGuess = neuralGuesses.indexOf(neuralGuesses.max()!!)
            if (bestGuess == correct) nCorrect++
        }
        return nCorrect
    }

    private fun svmLoss(xs: DoubleArray, correctIndex: Int): Double {
        val netOutput = this(xs)
        val correctScore = netOutput[correctIndex]
        return (0 until netOutput.size)
                .filter { it != correctIndex }
                .sumByDouble { Math.max(0.0, netOutput[it] - correctScore + 1.0) }
    }

    private fun softmax(netOutput: DoubleArray): DoubleArray {
        val max = netOutput.max() ?: 0.0
        for (i in 0 until netOutput.size) netOutput[i] -= max
        val sum = netOutput.sumByDouble { Math.exp(it) }
        return DoubleArray(netOutput.size) { i -> Math.exp(netOutput[i]) / sum }
    }

    private fun softmaxLoss(xs: DoubleArray, correctIndex: Int): Double {
        val netOutput = this(xs)
        val sm = softmax(netOutput)[correctIndex]
        return -Math.log(sm)
    }
}