package neural

class XorDataset : Dataset {
    override lateinit var xsTrainSplit: MutableList<List<DoubleArray>>
    override lateinit var ysTrainSplit: MutableList<List<Int>>
    override lateinit var xsTest: List<DoubleArray>
    override lateinit var ysTest: List<Int>
    override var nTraining: Int = -1
    override var numInputs: Int = 2
    override var numOutputs: Int = 2

    init {
        val trainingXs = listOf(doubleArrayOf(1.0, 1.0), doubleArrayOf(1.0, 0.0), doubleArrayOf(0.0, 1.0), doubleArrayOf(0.0, 0.0))
        val trainingYs = listOf(0, 1, 1, 0)
        nTraining = 4
        xsTest = listOf(doubleArrayOf(1.0, 1.0), doubleArrayOf(1.0, 0.0), doubleArrayOf(0.0, 1.0), doubleArrayOf(0.0, 0.0))
        ysTest = listOf(0, 1, 1, 0)
        splitTraining(trainingXs, trainingYs)
    }
}
