# Aloha VW JNI performance benchmark

Usage
```bash
sbt
> jmh:run -bm all
```
- Warmup iterations: 20
- Iterations: 20
- Running all benchmarks

### 

### Data generation
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
