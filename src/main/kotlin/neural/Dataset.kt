package neural

import java.util.concurrent.ThreadLocalRandom

data class Data(val trainingXs: List<DoubleArray>,
                val trainingYs: List<Int>)


abstract class Dataset {
    abstract fun getData(): Data
    lateinit var testXs: List<DoubleArray>
    lateinit var testYs: List<Int>
    abstract val numInputs: Int
    abstract val numOutputs: Int

    fun testAccuracy(best: Net): Double {
        var nCorrect = 0
        for ((i, testX) in testXs.withIndex()) {
            val correct = testYs[i]
            val neuralGuesses: DoubleArray = best(testX)
            val bestGuess = neuralGuesses.indexOf(neuralGuesses.max()!!)
            println("Net guess $bestGuess correct $correct")
            if (bestGuess == correct) nCorrect++
        }
        val accuracy = nCorrect.toDouble() / testXs.size
        println("Correct test-set classifications $nCorrect / ${testXs.size}")
        return accuracy
    }


    fun bootstrap(xs: MutableList<DoubleArray>, ys: MutableList<Int>) {
        val sizes = (0 until numOutputs).map { i -> ys.filter { it == i }.size }
        val max = sizes.max()!!

        for (i in 0 until numOutputs) {
            while (ys.filter { it == i }.size < max) {
                addOneBootstrapped(i, ys, xs)
            }
        }
    }

    private fun addOneBootstrapped(classIdx: Int, ys: MutableList<Int>, xs: MutableList<DoubleArray>) {
        val ids = ys.indices.filter { ys[it] == classIdx }
        // Get a feature from xs[ids]
        val newExample = DoubleArray(numInputs) { 0.0 }
        for (i in 0 until numInputs) {
            val r = ThreadLocalRandom.current().nextInt(ids.size)
            val id: Int = ids[r]
            val feature = xs[id][i]
            newExample[i] = feature
        }
        ys.add(classIdx)
        xs.add(newExample)
    }
}