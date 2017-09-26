class Specimen {
    private var data = ByteArray(0)
    private var target = ByteArray(0)
    var nCorrect: Long = 0
    var fitness: Long = 0

    constructor(target: ByteArray) {
        this.target = target
        this.data = ByteArray(target.size) { randomchar() }
        computeFitness()
    }

    private constructor(characters: ByteArray, fitness: Long, target: ByteArray) {
        this.data = characters
        this.fitness = fitness
        this.target = target
    }

    private fun computeCorrect() {
        nCorrect = 0
        for (i in 0 until data.size) if (target[i] == data[i]) nCorrect++
    }

    fun computeFitness() {
        computeCorrect()
        fitness = nCorrect * nCorrect
    }

    override fun toString(): String {
        return "Specimen(data='${byteArrayToString(data)}' fitness=$fitness)"
    }

    fun copy(): Specimen {
        return Specimen(this.data.copyOf(), this.fitness, this.target)
    }

    fun mutate(mutatefreq: Double) {
        for (i in 0 until data.size) if (random() < mutatefreq)
            data[i] = randomchar()
    }

    fun crossover(pool: List<Specimen>, crossoverfreq: Double) {
        val mate = Specimen.pick(pool)
        for (i in 0 until data.size) if (random() < crossoverfreq)
            data[i] = mate.data[i]
    }

    companion object {
        private val chars = ('a'..'z') + ' ' // + '.' + ',' + ('A'..'Z') +

        private fun randomchar(): Byte = chars[(random() * chars.size).toInt()].toByte()

        private var wheel = LongArray(0)

        fun computeWheel(arr: List<Specimen>) {
            var sum = 0L
            wheel = LongArray(arr.size) { i -> sum += arr[i].fitness; sum }
        }

        fun pick(arr: List<Specimen>): Specimen {
            val sum = wheel.last()
            val r = (random() * sum).toLong()
            var idx = wheel.binarySearch(r)
            if (idx < 0) idx = -idx - 1
            return arr[idx].copy()
        }
    }
}