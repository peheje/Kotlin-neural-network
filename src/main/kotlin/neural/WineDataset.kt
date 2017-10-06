package neural

import org.simpleflatmapper.csv.CsvParser
import java.io.FileReader
import java.util.*

class WineDataset : Dataset() {
    override val numInputs: Int
        get() = 13
    override val numOutputs: Int
        get() = 3

    override fun getData(): Data {

        // Read wine data
        val xs = mutableListOf<DoubleArray>()
        val ys = mutableListOf<Int>()

        CsvParser.stream(FileReader("datasets/wine.data")).use { stream ->
            stream.forEach { row ->
                xs.add(DoubleArray(13) { i -> row[i + 1].toDouble() })
                ys.add(row.first().toInt() - 1)    // Zero index it
            }
        }

        // Zero Normalize
        val nFeatures = xs.first().size
        for (featureId in 0 until nFeatures) {
            val avg = xs.sumByDouble { it[featureId] } / xs.size
            for (i in 0 until nFeatures) xs[featureId][i] -= avg
        }

        // Create training set and test set
        //val seed = System.nanoTime()
        val seed = 42L
        Collections.shuffle(xs, Random(seed))
        Collections.shuffle(ys, Random(seed))

        val total = 178
        val testSize = 10

        val trainingXs = xs.take(total-testSize)
        val trainingYs = ys.take(total-testSize)

        testXs = xs.takeLast(testSize)
        testYs = ys.takeLast(testSize)

        return Data(trainingXs, trainingYs)
    }
}