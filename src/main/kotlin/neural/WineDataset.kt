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
        val ys = mutableListOf<DoubleArray>()

        CsvParser.stream(FileReader("datasets/wine.data")).use { stream ->
            stream.forEach { row ->
                xs.add(DoubleArray(13) { i -> row[i + 1].toDouble() })
                ys.add(DoubleArray(1) { _ -> row.first().toDouble() - 1.0 })    // Zero index it
            }
        }

        // Create training set and test set
        val seed = System.nanoTime()
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