package neural

import java.util.*
import java.util.concurrent.ThreadLocalRandom

data class Data(val trainingXs: List<DoubleArray>,
                val trainingYs: List<Int>)


abstract class Dataset {
    abstract fun getData(): Data

    lateinit var xsSplit: MutableList<List<DoubleArray>>
    lateinit var ysSplit: MutableList<List<Int>>

    lateinit var testXs: List<DoubleArray>
    lateinit var testYs: List<Int>
    abstract val numInputs: Int
    abstract val numOutputs: Int

    fun split(xs: MutableList<DoubleArray>, ys: MutableList<Int>) {
        xsSplit = mutableListOf()
        ysSplit = mutableListOf()

        val classes = ys.distinct()
        for (c in classes) {
            val ids = ys.indices.filter { ys[it] == c }
            val xc = mutableListOf<DoubleArray>()
            val yc = mutableListOf<Int>()
            for (id in ids) {
                xc.add(xs[id])
                yc.add(ys[id])
            }
            xsSplit.add(xc)
            ysSplit.add(yc)
        }
    }

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

    fun zeroNormalize(xs: MutableList<DoubleArray>) {
        val nFeatures = xs.first().size
        for (featureId in 0 until nFeatures) {
            val avg = xs.sumByDouble { it[featureId] } / xs.size
            for (i in 0 until nFeatures) xs[featureId][i] -= avg
        }
    }

    fun shuffle(xs: MutableList<DoubleArray>, ys: MutableList<Int>) {
        // Create training set and test set
        val seed = System.nanoTime()
        //val seed = 86512L
        Collections.shuffle(xs, Random(seed))
        Collections.shuffle(ys, Random(seed))
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