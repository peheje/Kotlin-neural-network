package neural

import org.simpleflatmapper.csv.CsvParser
import java.io.FileReader
import java.util.*
import java.util.concurrent.ThreadLocalRandom

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

        bootstrap(xs, ys)
        zeroNormalize(xs)
        shuffle(xs, ys)

        val total = xs.size
        val testSize = 10

        val trainingXs = xs.take(total-testSize)
        val trainingYs = ys.take(total-testSize)

        testXs = xs.takeLast(testSize)
        testYs = ys.takeLast(testSize)

        return Data(trainingXs, trainingYs)
    }
}