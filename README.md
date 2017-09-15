# Aloha VW JNI performance benchmark

## Results

### Prediction Performance

It appears that, as expected, model performance goes down as more labels are
queried and seems to be (*mostly*) independent of the number of labels on which
the model was trained.

![Aloha VW JNI Performance Benchmark][throughput]

[throughput]: resources/aloha_perf.png

### VW vs Aloha Prediction Speed

It looks like as the number of labels queried increases, the amount of time
spent in an Aloha prediction becomes dominated by the actual VW prediction.

![Aloha vs VW JNI Performance][comparison]

[comparison]: resources/vw_over_aloha_speedup.png

### Raw Data

[Benchmark data][data] was generated on Ubuntu 16.04 on a [3930K](http://ark.intel.com/products/63697/Intel-Core-i7-3930K-Processor-12M-Cache-up-to-3_80-GHz)
using `jmh:run -wi 10 -i 10 -t 1 -f 1`.

[data]: resources/bench_data_3930k.tsv


## Usage
```bash
sbt
> jmh:run -wi 10 -i 10
```
- Warmup iterations: 10
- Iterations: 10
- Running throughput benchmark


## Data generation
```bash
FEATURES=$(seq 1 100 | sed 's/^/f/g' | tr '\n' ' ')
CLASSES=$(seq 1 10000 | sed 's/\(.*\)/\1:-1.0 |Y _C\1_/g')

# Construct one example and feed to VW.  
# Notice the ring_size needs to be about 10 more than the number of classes.
(cat <<EOD
shared |X $FEATURES
2147483648:0.0 |y _C2147483648_
2147483649:-1 |y _C2147483649_
$CLASSES

EOD
) | $VW \
 -b 23 \
 --ring_size 10010 \
 --csoaa_ldf mc \
 --csoaa_rank \
 --loss_function logistic \
 -q YX \
 --noconstant \
 --ignore_linear X \
 --ignore y \
 --initial_weight 0.000001 \
 --invert_hash /tmp/test.1.vw.hr.model \
 -f /tmp/test.1.vw.binary.model
 ```
