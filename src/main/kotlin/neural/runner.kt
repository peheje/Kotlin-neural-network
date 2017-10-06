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
    val mutatePower = 2.0
    val crossoverProp = 0.03
    val crossoverRate = 0.4
    val poolsize = 5000L
    val parentInheritance = 0.1
    val batchSize = 8
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
                timeInSeconds = 20,
                strategy = strategy,
                layerSetup = layerSetup,
                dataset = XorDataset()
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
                          layerSetup: List<Int>,
                          dataset: Dataset) {

    val x = mutableListOf<Double>()
    val y = mutableListOf<Double>()
    var generation = 0
    val origMutateProp = mutateProp
    val origMutatePower = mutatePower
    var mutateProp = mutateProp
    var mutatePower = mutatePower

    val (trainingXs, trainingYs) = dataset.getData()
//
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

    val best: Net = pool.maxBy { it.fitness }!!
    println(best)
    dataset.testAccuracy(best)
}