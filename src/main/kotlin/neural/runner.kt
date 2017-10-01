package neural

import koma.*
import random
import java.time.Duration
import java.time.Instant
import java.util.stream.Collectors.toList
import java.util.stream.Stream

fun neuralNetworkRunner() {

    val mutateProp = 0.35
    val mutateFreq = 0.25
    val mutateRate = 2.0
    val crossoverProp = 0.03
    val crossoverRate = 0.4
    val poolsize = 4000L
    val parentInheritance = 0.8
    val batchSize = 4

    //val mutateRates = linspace(0.40, 0.40, 1).toList()
    //val mutateProps = linspace(0.35, 0.35, 1).toList()
    //val mutateFreqs = linspace(0.25, 0.25, 1).toList()
    //val crossoverRates = linspace(0.4, 0.4, 1).toList()
    //val strategies = arrayOf(0)
    //val crossoverProps = linspace(0.04, 0.06, 3).toList()

    for ((color, _) in plotColors.keys.zip(listOf(0))) {
        geneticNeural(
                poolsize = poolsize,
                mutateProp = mutateProp,
                mutatePropDecay = 0.998,
                mutateFreq = mutateFreq,
                mutateRate = mutateRate,
                mutateRateDecay = 0.9995,
                crossoverProp = crossoverProp,
                crossoverRate = crossoverRate,
                parentInheritance = parentInheritance,
                batchSize = batchSize,
                plot = true,
                color = color,
                timeInSeconds = 60,
                strategy = 0
        )
    }
}


private fun geneticNeural(poolsize: Long,
                          mutateProp: Double,
                          mutatePropDecay: Double,
                          mutateFreq: Double,
                          mutateRate: Double,
                          mutateRateDecay: Double,
                          crossoverProp: Double,
                          crossoverRate: Double,
                          parentInheritance: Double,
                          batchSize: Int,
                          plot: Boolean,
                          color: String,
                          timeInSeconds: Int,
                          strategy: Int) {
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

    var t = -5.0
    while (t < 5.0) {
        trainingXsList.add(doubleArrayOf(t))
        if (t != 0.0)
            trainingYsList.add(doubleArrayOf(Math.sin(t) / t))
        else
            trainingYsList.add(doubleArrayOf(1.0))
        t += 0.01
    }

    val trainingXs = trainingXsList.toTypedArray()
    val trainingYs = trainingYsList.toTypedArray()

    // Algorithm go
    val starts = Instant.now()
    var pool = Stream.generate { Net(trainingXs, trainingYs, parentInheritance) }.parallel().limit(poolsize).collect(toList())
    while (Duration.between(starts, Instant.now()).seconds < timeInSeconds) {
        Net.computeWheel(pool)
        when (strategy) {
            0 -> {
                val nextGen = Stream.generate { Net.pick(pool) }.parallel().limit(poolsize).collect(toList())
                nextGen.parallelStream().forEach {
                    if (random() < crossoverProp) it.crossover(pool, crossoverRate)
                    if (random() < mutateProp) it.mutate(mutateFreq, mutateRate)
                    it.computeFitness(trainingXs, trainingYs, parentInheritance, batchSize)
                }
                pool = nextGen
                if (mutateRate > 0.02)
                    mutateRate *= mutateRateDecay
                if (mutateProp > 0.10)
                    mutateProp *= mutatePropDecay
            }
            1 -> {

            }
        }
        // Algorithm end

        if (generation++ % 100 == 0) {
            println("$generation: " + pool.maxBy { it.fitness })
            println("mutateProp $mutateProp mutateRate $mutateRate")
        }
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
                lineLabel = "bs $batchSize" +
                        "str $strategy" +
                        "mp $origMutateProp," +
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
    var i = -5.0
    while (i < 5.0) {
        val neuralGuess = best(doubleArrayOf(i)).first()
        val correct = if (i != 0.0)
            Math.sin(i) / i
        else
            1.0
        println("x: $i net: ${neuralGuess} true: ${correct} diff: ${neuralGuess - correct}")
        i += 0.5
    }
}