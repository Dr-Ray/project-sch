function linspace(start, stop, num, endpoint=true) {
    let x = [];
    x.push(start);
    if(endpoint) {
        let end = stop;
        x.push(stop);
        for (let i=0;i<num-2;i++) {
            end = ((start + end)/2);
            x.push(end);
        }
    }else{        
        if(num%2 == 0) {
            let bg = start, end = stop, mid = ((start + end)/2), mid2 = ((start + end)/2);
            x.push(mid);
            for (let i=0;i<num/2;i++) {
                mid = ((start + mid)/2);
                x.push(mid);
            }
            for (let i=0;i<num/2;i++) {
                end = ((mid2 + end)/2);
                x.push(end);
            }

        }else {
            let inc = start / num;
            let mid = start;
            for (let i=0;i<num-1;i++) {
                mid = mid + inc;
                x.push(mid);
            }
        }
        
    }

    return x;
}

let arr = linspace(1, 2, 4, false);

console.log(arr)