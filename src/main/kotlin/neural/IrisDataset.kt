package neural

import org.simpleflatmapper.csv.CsvParser
import java.io.FileReader
import java.util.*

class IrisDataset : Dataset() {
    override fun getData(): Data {
        // Read iris data
        val nameMap = mapOf(
                "Iris-virginica" to 0.0,
                "Iris-versicolor" to 1.0,
                "Iris-setosa" to 2.0)

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

        val trainingXs = xs.take(130)
        val trainingYs = ys.take(130)

        testXs = xs.takeLast(20)
        testYs = ys.takeLast(20)

        return Data(trainingXs, trainingYs)
    }
}