package neural

import org.simpleflatmapper.csv.CsvParser
import java.io.FileReader

class IrisDataset : Dataset {
    override lateinit var xsTrainSplit: MutableList<List<DoubleArray>>
    override lateinit var ysTrainSplit: MutableList<List<Int>>
    override lateinit var xsTest: List<DoubleArray>
    override lateinit var ysTest: List<Int>
    override var nTraining: Int = -1
    override var numInputs: Int = 4
    override var numOutputs: Int = 3

    init {
        // Read iris data
        val nameMap = mapOf(
                "Iris-virginica" to 0,
                "Iris-versicolor" to 1,
                "Iris-setosa" to 2)

        val xs = mutableListOf<DoubleArray>()
        val ys = mutableListOf<Int>()

        CsvParser.stream(FileReader("datasets/iris.data")).use { stream ->
            stream.forEach { row ->
                xs.add(DoubleArray(4) { i -> row[i].toDouble() })
                ys.add(nameMap[row.last()]!!)
            }
        }

        // Create training set and test set
        shuffle(xs, ys)

        nTraining = 130
        val trainingXs = xs.take(nTraining)
        val trainingYs = ys.take(nTraining)

        xsTest = xs.takeLast(20)
        ysTest = ys.takeLast(20)

        splitTraining(trainingXs.toMutableList(), trainingYs.toMutableList())
    }
}
