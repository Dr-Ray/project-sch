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

function range_(start, stop) {
    let arr = [];
    for(let i=0; i<stop; ++i) {
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
        if(body.method == "POST"){
            xhr.send(body.data);
        }else{
            xhr.send();
        }
    },

    fetch_request : (body, callback)=> {
        fetch(body.url, {
            headers: body.headers,
            method:body.method,
            body:body.data
        })
        .then(response=>response.json())
        .then(resp => callback(resp, null))
        .catch(err => callback(null, err))
    }
}