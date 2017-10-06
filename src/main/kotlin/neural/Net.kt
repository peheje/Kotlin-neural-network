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
        computeFitness(trainingXs, trainingYs, parentInheritance, gamma, batchSize = -1)
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

    fun computeFitness(trainingXs: List<DoubleArray>, trainingYs: List<Int>, parentInheritance: Double, gamma: Double, batchSize: Int) {
        var xs = trainingXs
        var ys = trainingYs
        val clippedBatchsize = Math.min(batchSize, xs.size)

        if (clippedBatchsize != -1) {
            val batchXs = mutableListOf<DoubleArray>()
            val batchYs = mutableListOf<Int>()
            while (batchXs.size < clippedBatchsize) {
                // Todo diversity in batch
                val r = ThreadLocalRandom.current().nextInt(xs.size)
                batchXs.add(xs[r])
                batchYs.add(ys[r])
            }
            xs = batchXs
            ys = batchYs
        }

        val dataLoss = (0 until xs.size).sumByDouble { svmLoss(xs[it], ys[it]) } / xs.size

        /*// Todo move to crossoverAndMutate?
        var regularizationLoss = 0.0
        for (layer in layers)
            for (neuron in layer.neurons)
                for (weight in neuron.weights)
                    regularizationLoss += weight*weight

        val loss = dataLoss + gamma * regularizationLoss*/

        val batchFitness = 1.0 / (dataLoss + 0.0001)
        val parentFitness = fitness
        fitness = parentFitness * parentInheritance + batchFitness
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
            for ((layerIdx, layer) in net.layers.withIndex()) {
                for ((neuronIdx, neuron) in layer.neurons.withIndex()) {
                    if (random() < crossoverProp) neuron.crossover(pool, layerIdx, neuronIdx, crossoverRate)
                    if (random() < mutateProp) neuron.mutate(mutatePower, mutateFreq)
                }
            }
        }
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
        throw Exception("please triple check implementation")
        val netOutput = this(xs)
        val sm = softmax(netOutput)[correctIndex]
        return -Math.log(sm)
    }
}