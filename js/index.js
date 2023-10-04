function addElement(tag, parent, optional={}){
    var elem = document.createElement(tag);
    if(optional){
        elem.setAttribute('id', optional.id)
        elem.setAttribute('class', optional.class)
    }
    parent.appendChild(elem);

    return elem;
}

function effect(elems, event, d_class) {
    for( let i=0; i<=elems.length; i++) {
        elems[i].addEventListener(event, ()=> {
            elems[i].classList.add(d_class);
        });
        elems[i].addEventListener(event, ()=> {
            elems[i].classList.remove(d_class);
        });
    }
}

function getElement(selector) {
    var pattern = /^#/;
    if(pattern.test(selector)) {
        return document.querySelector(selector);
    }else {
        return document.querySelectorAll(selector);
    }
}

function get_file_details(elem) {
    let {name, size, type} = elem, cnvt, typ;
    if(Math.round(size/1000000) < 0.1) {
        cnvt = Math.round(size/1000)+'kb';
    }else{
        cnvt = Math.round(size/1000000)+'mb';
    }

    typ = type.split('/')[1];

    return {name, "size": cnvt, "type": typ}
}

function range_(start, stop) {
    let arr = [];
    for(let i=start; i<stop; ++i) {
        arr.push(i);
    }
    return arr
}   

function getRndInteger(min, max) {
    return Math.floor(Math.random() * (max - min) ) + min;
}

var network = {
    xhr_request : (body, callback) => {
        var xhr = new XMLHttpRequest();
        xhr.open(body.method, body.url);
        if(body.content_type) {
            xhr.setRequestHeader("content-type", body.content_type);
        }
        xhr.onload = (e) => {
            if(xhr.status == 200) {
                var response = xhr.responseText;
                callback(response, null);
            }else {
                callback(null, xhr.responseText)
            }
        }
        if(body.method == "GET"){
            xhr.send();
        }else{
            xhr.send(body.data);
        }
    },

    fetch_request : async (body, callback)=> {
        try {
            const res = await fetch(body.url, {
                headers: body.headers,
                method:body.method,
                body:body.data
            });
            const resp = await res.json();
    
            return callback(resp, null);
        } catch (error) {
            return callback(null, error);
        }
    }
}