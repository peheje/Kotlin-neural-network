package neural

import java.util.*
import java.util.concurrent.ThreadLocalRandom

interface Dataset {
    var xsTrainSplit: MutableList<List<DoubleArray>>
    var ysTrainSplit: MutableList<List<Int>>
    var xsTest: List<DoubleArray>
    var ysTest: List<Int>
    var nTraining: Int
    var numInputs: Int
    var numOutputs: Int

    fun splitTraining(xs: List<DoubleArray>, ys: List<Int>) {
        xsTrainSplit = mutableListOf()
        ysTrainSplit = mutableListOf()

        val classes = ys.distinct()
        for (c in classes) {
            val ids = ys.indices.filter { ys[it] == c }
            val xc = mutableListOf<DoubleArray>()
            val yc = mutableListOf<Int>()
            for (id in ids) {
                xc.add(xs[id])
                yc.add(ys[id])
            }
            xsTrainSplit.add(xc)
            ysTrainSplit.add(yc)
        }
    }

    fun testAccuracy(best: Net): Double {
        var nCorrect = 0
        for ((i, testX) in xsTest.withIndex()) {
            val correct = ysTest[i]
            val neuralGuesses: DoubleArray = best(testX)
            val bestGuess = neuralGuesses.indexOf(neuralGuesses.max()!!)
            println("Net guess $bestGuess correct $correct")
            if (bestGuess == correct) nCorrect++
        }
        val accuracy = nCorrect.toDouble() / xsTest.size
        println("Correct test-set classifications $nCorrect / ${xsTest.size}")
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
        fun addOneBootstrapped(classIdx: Int, ys: MutableList<Int>, xs: MutableList<DoubleArray>) {
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

        val sizes = (0 until numOutputs).map { i -> ys.filter { it == i }.size }
        val max = sizes.max()!!

        for (i in 0 until numOutputs) {
            while (ys.filter { it == i }.size < max) {
                addOneBootstrapped(i, ys, xs)
            }
        }
    }
}