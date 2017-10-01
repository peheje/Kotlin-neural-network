package neural

import koma.*
import linspace
import random
import java.time.Duration
import java.time.Instant
import java.util.stream.Collectors

fun neuralNetworkRunner() {

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