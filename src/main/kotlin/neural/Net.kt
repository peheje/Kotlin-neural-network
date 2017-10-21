package neural

import random
import java.util.concurrent.ThreadLocalRandom

class Net {
    val layers: MutableList<Layer>
    private val nWeights: Int
    var fitness: Double = 0.0

    constructor(trainingXs: List<DoubleArray>, trainingYs: List<Int>, layerSetup: List<Int>, parentInheritance: Double, gamma: Double) {

        val nInput = layerSetup.first()
        val nOutput = layerSetup.last()
        var layerSetup = mutableListOf(nInput)
        val nLayers = ThreadLocalRandom.current().nextInt(1, 3)
        for (i in 0 until nLayers) {
            layerSetup.add(ThreadLocalRandom.current().nextInt(1, 8))
        }
        layerSetup.add(nOutput)

        nWeights = (0 until layerSetup.size - 1).sumBy { (1 + layerSetup[it]) * layerSetup[it + 1] }
        val lastLayerIdx = layerSetup.size - 2  // -2 as last is output size
        layers = (0 until layerSetup.size - 1).map { Layer(layerSetup[it], layerSetup[it + 1], it == lastLayerIdx) }.toMutableList()
        computeFitness(trainingXs, trainingYs, parentInheritance, gamma)
        fitness += ThreadLocalRandom.current().nextDouble(0.0, 1.0)
    }

    private constructor(layers: MutableList<Layer>, fitness: Double, nWeights: Int) {
        this.layers = layers
        this.fitness = fitness
        this.nWeights = nWeights
    }

    fun architecture(): String {
        val nInput = layers.first().neurons.first().weights.size
        // val nOutput = layers.last().neurons.size
        var arch: MutableList<Int> = layers.map { it.neurons.size }.toMutableList()
        arch.add(0, nInput)
        return arch.toString()
    }

    private fun copy(): Net {
        return Net(MutableList(layers.size) { i -> layers[i].copy() }, fitness, nWeights)
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
            for (layer in layers) for (neuron in layer.neurons) for (weight in neuron.weights) {
                regularizationLoss += weight * weight
            }
        }

        regularizationLoss = (regularizationLoss / nWeights.toDouble()) * gamma

        val dataLoss = (0 until xs.size).sumByDouble { softmaxLoss(xs[it], ys[it]) } / (xs.size + 0.0001)
        val correctFitness = nCorrectPredictions(xs, ys) / (xs.size + 0.0001)

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

        while (random() < crossoverRate) {
            // Take random weight from random neuron from random layer from mate
            val mateRanLayer = ThreadLocalRandom.current().nextInt(0, mate.layers.size)
            val mateRanNeuron = ThreadLocalRandom.current().nextInt(0, mate.layers[mateRanLayer].neurons.size)
            val mateRanWeight = ThreadLocalRandom.current().nextInt(0, mate.layers[mateRanLayer].neurons[mateRanNeuron].weights.size)

            // Take random weight from random neuron from random layer from me
            val ranLayer = ThreadLocalRandom.current().nextInt(0, layers.size)
            val ranNeuron = ThreadLocalRandom.current().nextInt(0, layers[ranLayer].neurons.size)
            val ranWeight = ThreadLocalRandom.current().nextInt(0, layers[ranLayer].neurons[ranNeuron].weights.size)

            // Lerp them
            val crossoverPower = ThreadLocalRandom.current().nextDouble(1.0)
            layers[ranLayer].neurons[ranNeuron].weights[ranWeight] = lerp(layers[ranLayer].neurons[ranNeuron].weights[ranWeight],
                    mate.layers[mateRanLayer].neurons[mateRanNeuron].weights[mateRanWeight], crossoverPower)
        }
    }

    fun crossover2(pool: List<Net>, crossoverRate: Double) {
        val mate = pick(pool)
        for ((layerIdx, layer) in layers.withIndex()) {
            for ((neuronIdx, neuron) in layer.neurons.withIndex()) {
                neuron.crossover(mate, layerIdx, neuronIdx, crossoverRate)
            }
        }
    }

    fun crossover3(pool: List<Net>, crossoverRate: Double) {
        val mate = pick(pool)
        for (i in 0 until layers.size)
            if (random() < crossoverRate) {
                val tmp = layers[i]
                layers[i] = mate.layers[i]
                mate.layers[i] = tmp
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

        internal fun lerp(a: Double, b: Double, p: Double): Double {
            return a + (b - a) * p
        }

        fun computeWheel(pool: List<Net>) {
            var sum = 0.0
            wheel = DoubleArray(pool.size) { i -> sum += pool[i].fitness; sum }
        }

        fun pick(pool: List<Net>): Net {
            val sum = wheel.last()
            val r = random() * sum
            var idx = wheel.binarySearch(r)
            if (idx < 0) idx = -idx - 1
            return pool[idx].copy()
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

        fun createBatch(dataset: Dataset, batchSize: Int): Pair<List<DoubleArray>, List<Int>> {

            // Todo Dataset
            val trainingSize = dataset.nTraining
            val clippedBatchSize = Math.min(batchSize, trainingSize)

            val batchXs = mutableListOf<DoubleArray>()
            val batchYs = mutableListOf<Int>()

            // Diverse batch as possible
            var c = 0
            while (batchXs.size < clippedBatchSize) {
                val r = ThreadLocalRandom.current().nextInt(dataset.xsTrainSplit[c].size)
                batchXs.add(dataset.xsTrainSplit[c][r])
                batchYs.add(dataset.ysTrainSplit[c][r])
                c = (c + 1) % dataset.numOutputs
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