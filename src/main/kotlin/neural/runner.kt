package neural

import koma.*
import random
import java.time.Duration
import java.time.Instant
import java.util.stream.Collectors.toList
import java.util.stream.Stream


fun neuralNetworkRunner() {

    val crossoverMutateRatio = 0.2

    val crossoverPower = 1.0

    val mutatePower = 1.10
    val mutatePowerDecay = 0.99

    val poolsize = 3_000L
    val parentDecay = 0.2
    val batchSize = 2
    val regularizationStrength = 0.0

    val dataset = WineDataset()
    val layerSetup = arrayListOf(dataset.numInputs, 8, 4, dataset.numOutputs)

    //val mutatePowers = linspace(0.40, 0.40, 1).toList()
    //val mutateProps = linspace(0.35, 0.35, 1).toList()
    //val mutateFreqs = linspace(0.25, 0.25, 1).toList()
    //val crossoverRates = linspace(0.4, 0.4, 1).toList()
    //val strategies = arrayOf(0)
    //val crossoverProps = linspace(0.04, 0.06, 3).toList()

    for ((color, strategy) in plotColors.keys.zip(listOf(0))) {
        geneticNeural(
                poolsize = poolsize,
                startMutatePower = mutatePower,
                mutatePowerDecay = mutatePowerDecay,
                crossoverPower = crossoverPower,
                parentDecay = parentDecay,
                gamma = regularizationStrength,
                batchSize = batchSize,
                plot = true,
                color = color,
                timeInSeconds = 30,
                strategy = strategy,
                layerSetup = layerSetup,
                dataset = dataset,
                crossoverMutateRatio = crossoverMutateRatio
        )
    }
}


private fun geneticNeural(poolsize: Long,
                          startMutatePower: Double,
                          mutatePowerDecay: Double,
                          crossoverPower: Double,
                          parentDecay: Double,
                          batchSize: Int,
                          gamma: Double,
                          plot: Boolean,
                          color: String,
                          timeInSeconds: Int,
                          strategy: Int,
                          layerSetup: List<Int>,
                          dataset: Dataset,
                          crossoverMutateRatio: Double) {

    val x = mutableListOf<Double>()
    val y = mutableListOf<Double>()
    var generation = 0
    var mutatePower = startMutatePower

    val (xs, ys) = dataset.getData()

    // Algorithm go
    val starts = Instant.now()
    var pool = Stream.generate { Net(xs, ys, layerSetup, parentDecay, gamma) }.parallel().limit(poolsize).collect(toList())
    while (Duration.between(starts, Instant.now()).seconds < timeInSeconds) {
        Net.computeWheel(pool)
        val nextGen = Stream.generate { Net.pick(pool) }.parallel().limit(poolsize).map {
            if (random() < crossoverMutateRatio) {
                val mate = it.crossover(pool, crossoverPower)
                it.calculateSexualFitness(mate, xs, ys, batchSize, parentDecay)
            }
            else {
                it.mutate(mutatePower)
                it.calculateAsexualFitness(xs, ys, batchSize, parentDecay)
            }
            it
        }.collect(toList())
        pool = nextGen

        if (mutatePower > 0.01)
            mutatePower *= mutatePowerDecay
        // Algorithm end

        if (generation++ % 100 == 0) {
            val currentBest = pool.maxBy { it.fitness }
            println("$generation: ${currentBest?.fitness} $currentBest")
            println(" mutatePower $mutatePower")
        }
        if (plot) {
            x.add(Duration.between(starts, Instant.now()).toMillis().toDouble())
            y.add(pool.maxBy { it.fitness }?.fitness ?: 0.0)
        }
    }

    if (plot) {
        figure(1)
        plotArrays(x.toDoubleArray(), y.toDoubleArray(), color,
                lineLabel = "bs $batchSize" +
                        " croMutRat $crossoverMutateRatio" +
                        "str $strategy" +
                        " cr $crossoverPower " +
                        " ps $poolsize" +
                        " mpo $startMutatePower"
        )
        xlabel("Miliseconds")
        ylabel("Fitness")
        title("Genetic algorithm")
    }

    val best: Net = pool.maxBy { it.fitness }!!
    println(best)
    dataset.testAccuracy(best)
}