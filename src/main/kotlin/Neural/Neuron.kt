package Neural

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