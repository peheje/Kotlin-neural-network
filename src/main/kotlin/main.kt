import neural.Net
import koma.*
import java.time.Duration
import java.time.Instant
import java.util.stream.Collectors

// BY peheje@github
// To run set settings.xml as your maven settings.xml (in file -> settings -> maven -> settings.xml override in IntelliJ)

fun main(args: Array<String>) {

    val mutateProp = 0.35
    val mutateFreq = 0.25
    val mutateRate = 0.40   // Should depend on training set relative size
    val crossoverProp = 0.02
    val crossoverRate = 0.4
    val poolsize = 5000

    val mutatePropDecays = linspace(0.9, 0.999, 4)
    val poolsizes = linspace(poolsize.toDouble(), poolsize.toDouble(), 1).toList()

    val mutateRates = linspace(0.40, 0.40, 1).toList()
    val mutateProps = linspace(0.35, 0.35, 1).toList()
    val mutateFreqs = linspace(0.25, 0.25, 1).toList()
    val crossoverProps = linspace(0.02, 0.02, 1).toList()
    val crossoverRates = linspace(0.4, 0.4, 1).toList()

    for (crossoverProp in crossoverProps) {
        for (crossoverRate in crossoverRates) {
            for (mutateProp in mutateProps) {
                for (mutateFreq in mutateFreqs) {
                    for ((color, mutateRate) in plotColors.keys.zip(mutateRates)) {
                        geneticNeural(
                                poolsize = poolsize,
                                mutateProp = mutateProp,
                                mutatePropDecay = 0.995,
                                mutateFreq = mutateFreq,
                                mutateRate = mutateRate,
                                crossoverProp = crossoverProp,
                                crossoverFreq = 0.1,
                                crossoverRate = crossoverRate,
                                plot = true,
                                color = color,
                                timeInSeconds = 60
                        )
                    }
                }
            }
        }
    }
}

private fun geneticNeural(poolsize: Int,
                          mutateProp: Double,
                          mutatePropDecay: Double,
                          mutateFreq: Double,
                          mutateRate: Double,
                          crossoverProp: Double,
                          crossoverFreq: Double,
                          crossoverRate: Double,
                          plot: Boolean,
                          color: String,
                          timeInSeconds: Int) {
    val x = mutableListOf<Double>()
    val y = mutableListOf<Double>()
    var generation = 0
    val origMutateProp = mutateProp
    val origMutateRate = mutateRate
    var mutateProp = mutateProp
    var mutateRate = mutateRate

    /*
    // Learn XOR
    val trainingXs = arrayOf(doubleArrayOf(1.0, 1.0), doubleArrayOf(1.0, 0.0), doubleArrayOf(0.0, 1.0), doubleArrayOf(0.0, 0.0))
    val trainingYs = arrayOf(doubleArrayOf(0.0), doubleArrayOf(1.0), doubleArrayOf(1.0), doubleArrayOf(0.0))
    */

    // Learn AND
    /*
    val trainingXs = arrayOf(doubleArrayOf(1.0, 1.0), doubleArrayOf(1.0, 0.0), doubleArrayOf(0.0, 1.0), doubleArrayOf(0.0, 0.0))
    val trainingYs = arrayOf(doubleArrayOf(1.0), doubleArrayOf(0.0), doubleArrayOf(0.0), doubleArrayOf(0.0))
    */

    val trainingXsList = mutableListOf<DoubleArray>()
    val trainingYsList = mutableListOf<DoubleArray>()

    var t = -10.0
    while (t < 10.0) {
        trainingXsList.add(doubleArrayOf(t))
        trainingYsList.add(doubleArrayOf(Math.sin(t)/t))
        t += 0.01
    }

    val trainingXs = trainingXsList.toTypedArray()
    val trainingYs = trainingYsList.toTypedArray()

    // Algorithm go
    val starts = Instant.now()
    var pool = List(poolsize) { Net(trainingXs, trainingYs) }
    while (Duration.between(starts, Instant.now()).seconds < timeInSeconds) {
        Net.computeWheel(pool)
        val poolList = pool.parallelStream().map { Net.pick(pool) }.collect(Collectors.toList())
        poolList.parallelStream().forEach {
            if (random() < crossoverProp) it.crossover(pool, crossoverFreq, crossoverRate)
            if (random() < mutateProp) it.mutate(mutateFreq, mutateRate)
            it.computeFitness(trainingXs, trainingYs, 16)
        }
        pool = poolList
        mutateProp *= mutatePropDecay
        // Algorithm end

        if (generation++ % 100 == 0) println("$generation: " + pool.maxBy { it.fitness })
        if (plot) {
            x.add(Duration.between(starts, Instant.now()).toMillis().toDouble())
            y.add(pool.maxBy { it.fitness }?.fitness ?: 0.0)
        }
    }

    val best = pool.maxBy { it.fitness }!!
    println(best)

    if (plot) {
        figure(1)

        plotArrays(x.toDoubleArray(), y.toDoubleArray(), color,
                lineLabel = "mp $origMutateProp," +
                        " cr $crossoverRate " +
                        " cp $crossoverProp" +
                        " mf $mutateFreq" +
                        " mpd $mutatePropDecay" +
                        " ps $poolsize" +
                        " mr $origMutateRate"
        )

        xlabel("Miliseconds")
        ylabel("Fitness")
        title("Genetic algorithm")
    }

    // Test for XOR and AND
    /*
    for ((i, x) in trainingXs.withIndex()) {
        val neuralGuess = best(x)
        val correct = trainingYs[i]
        println("Net guessed ${neuralGuess.toList()} true was ${correct.toList()}")
    }
    */

    // Test for mathematical
    var i = -10.1
    while (i < 10.0) {
        val neuralGuess = best(doubleArrayOf(i))
        val correct = Math.sin(i)/i
        println("x: $i net: ${neuralGuess.toList()} true: ${correct}")
        i += 1.0
    }
}

fun genetic() {
    val mutateProp = 0.10    // Prop that specimen will mutate
    val mutatePropDecay = 0.95
    val mutateFreq = 0.04    // Prop that character will mutate

    val crossoverProp = 0.48 // Prop that specimen will crossover
    val crossoverFreq = 0.37 // Prop that character will crossover

    // val target = stringToByteArray("to be or not to be that is the question")
    // val target = stringToByteArray("the time when the computer was a gray and tiresome box on the floor is a long time ago a longer time ago than far far away in a galaxy of the guardians")
    val target = stringToByteArray("the hazards of visiting this island were legendary it was not just the hostility of the best anchorage on the island nor the odd accidents known to befall ships and visitors the whole of the island was enshrouded in the peculiar magic of the others kennit had felt it tugging at him as he and gankis followed the path that led from deception cove to the treasure beach for a path seldom used its black gravel was miraculously clean of fallen leaves or intruding plant life about them the trees dripped the secondhand rain of last nights storm onto fern fronds already burdened with crystal drops")
    val poolsize = 10_000         // Poolsize

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


