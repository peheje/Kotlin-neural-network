import koma.*
import java.time.Duration
import java.time.Instant
import java.util.concurrent.ThreadLocalRandom
import java.util.stream.Collectors

fun Double.format(digits: Int) = java.lang.String.format("%.${digits}f", this)

// BY peheje@github
// To run set settings.xml as your maven settings.xml (in file -> settings -> maven -> settings.xml (override)

fun stringToByteArray(str: String): ByteArray {
    return kotlin.ByteArray(str.length) {i -> str[i].toByte()}
}

fun byteArrayToString(bytes: ByteArray): String {
    return Array(bytes.size) { i -> bytes[i].toChar() }.joinToString ("")
}

fun random(): Double {
    return ThreadLocalRandom.current().nextDouble()
}

class Specimen {
    private var data = ByteArray(0)
    private var target = ByteArray(0)
    var nCorrect: Long = 0
    var fitness: Long = 0

    constructor(target: ByteArray) {
        this.target = target
        this.data = ByteArray(target.size, { randomchar().toByte() })
        computeFitness()
    }

    private constructor(characters: ByteArray, fitness: Long, target: ByteArray) {
        this.data = characters
        this.fitness = fitness
        this.target = target
    }

    private fun calculateCorrect() {
        nCorrect = 0
        for (i in 0 until data.size) if (target[i] == data[i]) nCorrect++
    }

    fun computeFitness() {
        calculateCorrect()
        fitness = nCorrect * nCorrect
    }

    override fun toString(): String {
        return "Specimen(data='${byteArrayToString(data)}' fitness=$fitness)"
    }

    fun copy(): Specimen {
        return Specimen(this.data.copyOf(), this.fitness, this.target)
    }

    fun mutate(mutatefreq: Double) {
        for (i in 0 until data.size) if (random() < mutatefreq)
            data[i] = randomchar()
    }

    fun crossover(pool: List<Specimen>, crossoverfreq: Double) {
        val mate = Specimen.pick(pool)
        for (i in 0 until data.size) if (random() < crossoverfreq)
            data[i] = mate.data[i]
    }

    companion object {
        private val chars = ('a'..'z') + ' '

        private fun randomchar(): Byte = chars[(random() * chars.size).toInt()].toByte()

        private var wheel = LongArray(0)

        fun computeWheel(arr: List<Specimen>) {
            var sum = 0L
            wheel = LongArray(arr.size, { i -> sum += arr[i].fitness; sum })
        }

        fun pick(arr: List<Specimen>): Specimen {
            val sum = wheel.last()
            val r = (random() * sum).toLong()
            var idx = wheel.binarySearch(r)
            if (idx < 0) idx = -idx - 1
            return arr[idx].copy()
        }
    }
}

fun main(args: Array<String>) {
    genetic()
}

fun genetic() {
    val mutateProp = 0.07    // Prop that specimen will mutate
    val mutatePropDecay = 0.9
    val mutateFreq = 0.04    // Prop that character will mutate

    val crossoverProp = 0.48 // Prop that specimen will crossover
    val crossoverFreq = 0.37 // Prop that character will crossover

    // val target = stringToByteArray("to be or not to be that is the question")
    // val target = stringToByteArray("the time when the computer was a gray and tiresome box on the floor is a long time ago a longer time ago than far far away in a galaxy of the guardians")
    val target = stringToByteArray("the hazards of visiting this island were legendary it was not just the hostility of the best anchorage on the island nor the odd accidents known to befall ships and visitors the whole of the island was enshrouded in the peculiar magic of the others kennit had felt it tugging at him as he and gankis followed the path that led from deception cove to the treasure beach for a path seldom used its black gravel was miraculously clean of fallen leaves or intruding plant life about them the trees dripped the secondhand rain of last night's storm onto fern fronds already burdened with crystal drops")
    val poolsize = 5_000         // Poolsize

    val plot = true
    val timeInSeconds = 20

    val colors = plotColors.keys
    val selectionStrategy = listOf(0)
    val poolsizes = linspace(10_000.0, 10_000.0, 1).toList()
    val crossoverProps = linspace(0.1, 0.9, 3)
    val crossoverFreqs = linspace(0.1, 0.9, 3).toList()
    for ((color, strategy) in colors.zip(selectionStrategy))
        geneticAlgorithm(poolsize, target, mutateProp, mutatePropDecay, mutateFreq, crossoverProp, crossoverFreq, plot, color, timeInSeconds, strategy)
}

private fun geneticAlgorithm(poolsize: Int, targetString: ByteArray, mutateProp: Double, mutatePropDecay: Double, mutateFreq: Double, crossoverProp: Double, crossoverFreq: Double, plot: Boolean, color: String, timeInSeconds: Int, strategy: Int) {
    val x = mutableListOf<Double>()
    val y = mutableListOf<Double>()
    var generation = 0
    var mutateProp = mutateProp

    // Algorithm go
    val starts = Instant.now()
    var pool = List(poolsize) { Specimen(targetString) }
    while (Duration.between(starts, Instant.now()).seconds < timeInSeconds) {
        when (strategy) {
            0 -> {
                Specimen.computeWheel(pool)
                val poolList = pool.parallelStream().map { Specimen.pick(pool) }.collect(Collectors.toList())
                poolList.parallelStream().forEach {
                    if (random() < crossoverProp) it.crossover(pool, crossoverFreq)
                    if (random() < mutateProp) it.mutate(mutateFreq)
                    it.computeFitness()
                }
                pool = poolList
            }
        }
        mutateProp *= mutatePropDecay
        // Algorithm end

        if (generation++ % 100 == 0) println("$generation: " + pool.maxBy { it.fitness })
        if (plot) {
            x.add(Duration.between(starts, Instant.now()).toMillis().toDouble())
            y.add(pool.maxBy { it.nCorrect }?.nCorrect?.toDouble() ?: 0.0)
        }
    }
    println(pool.maxBy { it.fitness })

    if (plot) {
        figure(1)
        plotArrays(x.toDoubleArray(), y.toDoubleArray(), color, lineLabel = "st: $strategy, cr-fr: ${crossoverFreq.format(3)}, cr-pr: ${crossoverProp.format(3)}")
        xlabel("Miliseconds")
        ylabel("Correct characters")
        title("Genetic algorithm")
    }
}


fun linspace(min: Double, max: Double, points: Int): DoubleArray {
    val d = DoubleArray(points)
    val step = (max - min) / (points - 1)
    for (i in 0 until points) {
        d[i] = min + i * step
    }
    return d
}

fun map(x: Double, originFrom: Double, originTo: Double, from: Double, to: Double): Double {
    //Y = (X-A)/(B-A) * (D-C) + C
    return (x - originFrom) / (originTo - originFrom) * (to - from) + from
}

fun softmax(x: DoubleArray): DoubleArray {
    val max = x.max() ?: 0.0
    for (i in 0 until x.size) x[i] -= max
    val sum = x.sumByDouble { Math.exp(it) }
    return DoubleArray(x.size) { i -> Math.exp(x[i]) / sum }
}