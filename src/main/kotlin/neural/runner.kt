package neural

import koma.*
import org.simpleflatmapper.csv.CsvParser
import random
import java.io.FileReader
import java.time.Duration
import java.time.Instant
import java.util.*
import java.util.stream.Collectors.toList
import java.util.stream.Stream


fun neuralNetworkRunner() {
    val mutateProp = 0.35
    val mutateFreq = 0.25
    val mutatePower = 2.0
    val crossoverProp = 0.03
    val crossoverRate = 0.4
    val poolsize = 10_000L
    val parentInheritance = 0.9
    val batchSize = 4
    val layerSetup = arrayListOf(4, 8, 4, 3)

    //val mutatePowers = linspace(0.40, 0.40, 1).toList()
    //val mutateProps = linspace(0.35, 0.35, 1).toList()
    //val mutateFreqs = linspace(0.25, 0.25, 1).toList()
    //val crossoverRates = linspace(0.4, 0.4, 1).toList()
    //val strategies = arrayOf(0)
    //val crossoverProps = linspace(0.04, 0.06, 3).toList()

    for ((color, strategy) in plotColors.keys.zip(listOf(0))) {
        geneticNeural(
                poolsize = poolsize,
                mutateProp = mutateProp,
                mutatePropDecay = 0.9995,
                mutateFreq = mutateFreq,
                mutatePower = mutatePower,
                mutatePowerDecay = 0.9997,
                crossoverProp = crossoverProp,
                crossoverRate = crossoverRate,
                parentInheritance = parentInheritance,
                batchSize = batchSize,
                plot = true,
                color = color,
                timeInSeconds = 10,
                strategy = strategy,
                layerSetup = layerSetup
        )
    }
}


private fun geneticNeural(poolsize: Long,
                          mutateProp: Double,
                          mutatePropDecay: Double,
                          mutateFreq: Double,
                          mutatePower: Double,
                          mutatePowerDecay: Double,
                          crossoverProp: Double,
                          crossoverRate: Double,
                          parentInheritance: Double,
                          batchSize: Int,
                          plot: Boolean,
                          color: String,
                          timeInSeconds: Int,
                          strategy: Int,
                          layerSetup: List<Int>) {
    val x = mutableListOf<Double>()
    val y = mutableListOf<Double>()
    var generation = 0
    val origMutateProp = mutateProp
    val origMutatePower = mutatePower
    var mutateProp = mutateProp
    var mutatePower = mutatePower


    // Learn XOR
    /*
    val trainingXs = arrayOf(doubleArrayOf(1.0, 1.0), doubleArrayOf(1.0, 0.0), doubleArrayOf(0.0, 1.0), doubleArrayOf(0.0, 0.0))
    val trainingYs = arrayOf(doubleArrayOf(0.0), doubleArrayOf(1.0), doubleArrayOf(1.0), doubleArrayOf(0.0))
    */

    // Learn AND
    /*
    val trainingXs = arrayOf(doubleArrayOf(1.0, 1.0), doubleArrayOf(1.0, 0.0), doubleArrayOf(0.0, 1.0), doubleArrayOf(0.0, 0.0))
    val trainingYs = arrayOf(doubleArrayOf(1.0), doubleArrayOf(0.0), doubleArrayOf(0.0), doubleArrayOf(0.0))
    */

    // Learning sin(x)/x
    /*
    val trainingXsList = mutableListOf<DoubleArray>()
    val trainingYsList = mutableListOf<DoubleArray>()

    var t = -5.0
    while (t < 5.0) {
        trainingXsList.add(doubleArrayOf(t))
        if (t != 0.0)
            trainingYsList.add(doubleArrayOf((Math.sin(t) / t) * 1.0))
        else
            trainingYsList.add(doubleArrayOf(1.0))
        t += 0.01
    }

    val trainingXs = trainingXsList.toTypedArray()
    val trainingYs = trainingYsList.toTypedArray()
    */


    // Read iris data
    val nameMap = mapOf("Iris-virginica" to 0.0, "Iris-versicolor" to 1.0, "Iris-setosa" to 2.0)
    val xs = mutableListOf<DoubleArray>()
    val ys = mutableListOf<DoubleArray>()
    CsvParser.stream(FileReader("datasets/iris.data")).use { stream ->
        stream.forEach { row ->
            xs.add(DoubleArray(4) { i -> row[i].toDouble() })
            ys.add(DoubleArray(1) { _ -> nameMap[row.last()]!! })
        }
    }

    // Create training set and test set
    val seed = System.nanoTime()
    Collections.shuffle(xs, Random(seed))
    Collections.shuffle(ys, Random(seed))

    val trainingXs = xs.take(140)
    val trainingYs = ys.take(140)

    val testXs = xs.takeLast(10)
    val testYs = ys.takeLast(10)

    // Algorithm go
    val starts = Instant.now()
    var pool = Stream.generate { Net(trainingXs, trainingYs, layerSetup, parentInheritance) }.parallel().limit(poolsize).collect(toList())
    while (Duration.between(starts, Instant.now()).seconds < timeInSeconds) {
        Net.computeWheel(pool)
        val nextGen = Stream.generate { Net.pick(pool) }.parallel().limit(poolsize).map {
            if (random() < crossoverProp) it.crossover(pool, crossoverRate)
            if (random() < mutateProp) it.mutate(mutateFreq, mutatePower)
            it.computeFitness(trainingXs, trainingYs, parentInheritance, batchSize)
            it
        }.collect(toList())
        pool = nextGen
        if (mutatePower > 0.10)
            mutatePower *= mutatePowerDecay
        if (mutateProp > 0.05)
            mutateProp *= mutatePropDecay
        // Algorithm end

        if (generation++ % 100 == 0) {
            println("$generation: " + pool.maxBy { it.fitness })
            println("mutateProp $mutateProp mutatePower $mutatePower")
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
                        " mp $origMutatePower"
        )
        xlabel("Miliseconds")
        ylabel("Fitness")
        title("Genetic algorithm")
    }

    // Test for iris

    var nCorrect = 0
    for ((i, testX) in testXs.withIndex()) {

        val correct = Math.round(testYs[i].first()).toInt()
        val neuralGuesses: DoubleArray = best(testX)
        val bestGuess = neuralGuesses.indexOf(neuralGuesses.max()!!)
        println("Net guess $bestGuess correct $correct")
        if (bestGuess == correct) nCorrect++
    }
    println("Correct iris test-set classifications $nCorrect / ${testXs.size}")


    // Test for XOR and AND

    /*
    for ((i, x) in trainingXs.withIndex()) {
        val neuralGuess = best(x)
        val correct = trainingYs[i]
        println("Net guessed ${neuralGuess.toList()} true was ${correct.toList()}")
    }
    */

    /*
    // Test for mathematical
    var i = -5.0
    while (i < 5.0) {
        val neuralGuess = best(doubleArrayOf(i)).first()
        val correct = if (i != 0.0)
            (Math.sin(i) / i) * 1.0
        else
            1.0
        println("x: \t $i net: ${neuralGuess} \t true: ${correct} \t diff: ${neuralGuess - correct}")
        i += 0.5
    }
    */
}