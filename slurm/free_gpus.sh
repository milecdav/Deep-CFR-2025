#!/usr/bin/env bash

scontrol show nodes | awk '

function add(part, total, used) {
    total_gpu[part] += total
    used_gpu[part]  += used
}

{
    # Partitions
    if ($0 ~ /Partitions=/) {
        part_line = $0
        sub(/.*Partitions=/, "", part_line)
        sub(/ .*/, "", part_line)
        split(part_line, plist, ",")
    }

    # Total GPUs
    if ($0 ~ /Gres=gpu/) {
        total = 0
        if (match($0, /gpu:[^:]*:[0-9]+/)) {
            gpu_str = substr($0, RSTART, RLENGTH)
            split(gpu_str, tmp, ":")
            total = tmp[3]
        }
    }

    # Used GPUs
    if ($0 ~ /AllocTRES=/) {
        used = 0
        if (match($0, /gres\/gpu=[0-9]+/)) {
            used_str = substr($0, RSTART, RLENGTH)
            split(used_str, tmp2, "=")
            used = tmp2[2]
        }

        for (i in plist) {
            add(plist[i], total, used)
        }
    }
}

END {
    printf "\nGPU availability by partition:\n"
    for (p in total_gpu) {
        if (total_gpu[p] > 0) {
            free = total_gpu[p] - used_gpu[p]
            perc = (free / total_gpu[p]) * 100
            printf "%-20s %4d / %-4d GPUs free (%5.1f%%)\n",
                   p ":", free, total_gpu[p], perc
        }
    }
    print ""
}
'