package neural

class XorDataset : Dataset() {
    override val numInputs: Int
        get() = 2
    override val numOutputs: Int
        get() = 2

    override fun getData(): Data {
        val trainingXs = listOf(doubleArrayOf(1.0, 1.0), doubleArrayOf(1.0, 0.0), doubleArrayOf(0.0, 1.0), doubleArrayOf(0.0, 0.0))
        val trainingYs = listOf(doubleArrayOf(0.0), doubleArrayOf(1.0), doubleArrayOf(1.0), doubleArrayOf(0.0))

        testXs = listOf(doubleArrayOf(1.0, 1.0), doubleArrayOf(1.0, 0.0), doubleArrayOf(0.0, 1.0), doubleArrayOf(0.0, 0.0))
        testYs = listOf(doubleArrayOf(0.0), doubleArrayOf(1.0), doubleArrayOf(1.0), doubleArrayOf(0.0))

        return Data(trainingXs, trainingYs)
    }
}