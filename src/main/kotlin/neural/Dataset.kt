package neural

data class Data(val trainingXs: List<DoubleArray>,
                val trainingYs: List<DoubleArray>)


abstract class Dataset {
    abstract fun getData(): Data
    lateinit var testXs: List<DoubleArray>
    lateinit var testYs: List<DoubleArray>

    fun testAccuracy(best: Net): Double {
        var nCorrect = 0
        for ((i, testX) in testXs.withIndex()) {
            val correct = Math.round(testYs[i].first()).toInt()
            val neuralGuesses: DoubleArray = best(testX)
            val bestGuess = neuralGuesses.indexOf(neuralGuesses.max()!!)
            println("Net guess $bestGuess correct $correct")
            if (bestGuess == correct) nCorrect++
        }
        val accuracy = nCorrect.toDouble() / testXs.size
        println("Correct test-set classifications $nCorrect / ${testXs.size}")
        return accuracy
    }
}