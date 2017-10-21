package neural

import org.simpleflatmapper.csv.CsvParser
import java.io.FileReader

class WineDataset : Dataset {
    override lateinit var xsTrainSplit: MutableList<List<DoubleArray>>
    override lateinit var ysTrainSplit: MutableList<List<Int>>
    override lateinit var xsTest: List<DoubleArray>
    override lateinit var ysTest: List<Int>
    override var nTraining: Int = -1
    override var numInputs: Int = 13
    override var numOutputs: Int = 3

    init {
        val xs = mutableListOf<DoubleArray>()
        val ys = mutableListOf<Int>()
        CsvParser.stream(FileReader("datasets/wine.data")).use { stream ->
            stream.forEach { row ->
                xs.add(DoubleArray(13) { i -> row[i + 1].toDouble() })
                ys.add(row.first().toInt() - 1)    // Zero index it
            }
        }
        shuffle(xs, ys)
        val total = xs.size
        val testSize = 10
        nTraining = total-testSize
        val trainingXs = xs.take(total-testSize)
        val trainingYs = ys.take(total-testSize)
        xsTest = xs.takeLast(testSize)
        ysTest = ys.takeLast(testSize)
        bootstrap(xs, ys)
        zeroNormalize(xs)
        splitTraining(trainingXs, trainingYs)
    }

}