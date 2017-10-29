package neural

import koma.*
import linspace
import random
import java.time.Duration
import java.time.Instant
import java.util.stream.Collectors.toList
import java.util.stream.Stream


fun neuralNetworkRunner() {
    val mutateProp = 0.10           // Probability of a net mutating at all each generation
    val mutatePropDecay = 1.0       // Decay rate of mutateProp
    val mutateFreq = 0.10           // Probability that a weight in a net to be mutated is mutated
    val mutatePower = 0.5           // The power [-mutatePower, mutatePower] of mutation
    val mutatePowerDecay = 1.0      // Decay rate of mutatePower
    val smoothing = 0.0

    val crossoverProp = 0.1         // Probability of a net crossing over each generation
    val crossoverRate = 0.05        // Ratio of number of weights to crossover when crossing over

    val dataset = WineDataset()
    val poolsize = 10_000L
    val batchSize = Math.max(dataset.numOutputs, 4)
    val parentInheritance = 0.8
    val regularizationStrength = 0.02

    val randomArchitecture = true
    val layerSetup = arrayListOf(dataset.numInputs, 10, 4, dataset.numOutputs)

    //val mutatePowers = linspace(0.40, 0.40, 1).toList()
    //val mutateProps = linspace(0.35, 0.35, 1).toList()
    //val mutateFreqs = linspace(0.25, 0.25, 1).toList()
    //val crossoverRates = linspace(0.4, 0.4, 1).toList()
    //val strategies = arrayOf(0)
    //val crossoverProps = linspace(0.04, 0.06, 3).toList()

    for ((color, _) in plotColors.keys.zip(linspace(0.0, 0.5, 1).toList())) {
        geneticNeural(
                poolsize = poolsize,
                startMutateProp = mutateProp,
                mutatePropDecay = mutatePropDecay,
                mutateFreq = mutateFreq,
                startMutatePower = mutatePower,
                mutatePowerDecay = mutatePowerDecay,
                crossoverProp = crossoverProp,
                crossoverRate = crossoverRate,
                parentInheritance = parentInheritance,
                batchSize = batchSize,
                gamma = regularizationStrength,
                plot = true,
                color = color,
                timeInSeconds = 20,
                strategy = 0,
                layerSetup = layerSetup,
                dataset = dataset,
                smoothing = smoothing,
                randomArhictecture = randomArchitecture
        )
    }
}


private fun geneticNeural(poolsize: Long,
                          startMutateProp: Double,
                          mutatePropDecay: Double,
                          mutateFreq: Double,
                          startMutatePower: Double,
                          mutatePowerDecay: Double,
                          crossoverProp: Double,
                          crossoverRate: Double,
                          parentInheritance: Double,
                          batchSize: Int,
                          gamma: Double,
                          plot: Boolean,
                          color: String,
                          timeInSeconds: Int,
                          strategy: Int,
                          layerSetup: List<Int>,
                          dataset: Dataset,
                          smoothing: Double,
                          randomArhictecture: Boolean) {

    val x = mutableListOf<Double>()
    val y = mutableListOf<Double>()
    val yBests = mutableListOf<Double>()
    var generation = 0
    var mutateProp = startMutateProp
    var mutatePower = startMutatePower

    val trainingXs = dataset.xsTrainSplit.flatMap { it }
    val trainingYs = dataset.ysTrainSplit.flatMap { it }

    // Algorithm go
    val starts = Instant.now()
    var pool = Stream.generate { Net(trainingXs, trainingYs, layerSetup, parentInheritance, gamma, randomArhictecture) }.parallel().limit(poolsize).collect(toList())
    while (Duration.between(starts, Instant.now()).seconds < timeInSeconds) {
        Net.computeWheel(pool)
        val (bxs, bys) = Net.createBatch(dataset, batchSize)
        val nextGen = Stream.generate { Net.pick(pool) }.parallel().limit(poolsize - 1).map {
            if (random() < crossoverProp) it.crossover(pool, crossoverRate)
            if (random() < mutateProp) it.mutate(mutateFreq, mutatePower)
            it.computeFitness(bxs, bys, parentInheritance, gamma)
            it
        }.collect(toList())

        pool = nextGen

        if (mutatePower > 0.10)
            mutatePower *= mutatePowerDecay
        if (mutateProp > 0.05)
            mutateProp *= mutatePropDecay
        // Algorithm end

        if (generation++ % 100 == 0) {
            val currentBest = pool.maxBy { it.fitness }
            println("$generation: ${currentBest?.fitness} $currentBest")
            println("mutateProp $mutateProp mutatePower $mutatePower")
        }
        if (plot) {
            x.add(Duration.between(starts, Instant.now()).toMillis().toDouble())
            val best = pool.maxBy { it.fitness }!!

            if (smoothing == 0.0)
                y.add(dataset.testAccuracy(best, false))
            else {
                val smoothingN = (smoothing * poolsize).toInt()
                y.add(yBests.take(smoothingN).sum()/smoothingN)
                yBests.add(best.fitness)
            }
        }
    }

    if (plot) {
        figure(1)
        plotArrays(x.toDoubleArray(), y.toDoubleArray(), color,
                lineLabel = "bs $batchSize" +
                        "str $strategy" +
                        " mpr $startMutateProp," +
                        " cr $crossoverRate " +
                        " cp $crossoverProp" +
                        " mf $mutateFreq" +
                        " mpd $mutatePropDecay" +
                        " ps $poolsize" +
                        " mpo $startMutatePower"
        )
        xlabel("Miliseconds")
        ylabel("Fitness")
        title("Genetic algorithm")
    }

    /*
    val best: Net = pool.maxBy { it.fitness }!!
    dataset.testAccuracy(best)
    println("the best was: $best")
    println("the best had architecture: ${best.architecture()}")
    */

    val bests: List<Net> = pool.sortedBy { it.fitness }.takeLast(10)
    dataset.testAccuracy(bests)
}