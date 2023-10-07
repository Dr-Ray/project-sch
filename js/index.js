function addElement(tag, parent, optional={}){
    var elem = document.createElement(tag);
    if(optional){
        elem.setAttribute('id', optional.id)
        elem.setAttribute('class', optional.class)
    }
    parent.appendChild(elem);

    return elem;
}

function _event(elems, event, d_class) {
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
            xhr.setRequestHeader("content-Type", body.content_type);
        }
        xhr.onload = (e) => {
            if(xhr.status == 200) {
                // getElement("#anim_loader").classList.add('rm-loader');
                var response = xhr.responseText;
                callback(response, null);
            }else {
                callback(null, xhr.responseText)
            }
        }
        if(body.data) {
            if(body.method == "GET"){
                xhr.send();
            }else{
                xhr.send(body.data);
            }
        }else{
            xhr.send();
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

function createLoader(img='', text='', color='#000',) {
    var elem = addElement('div', getElement('#r-app'),{
        "id":"anim_loader",
        "class":"loader"
    });

    elem.style.background = `linear-gradient(rgba(0,0,0,.5),rgba(0,0,0,.5)), url("./images/${img}")`;
    elem.style.backgroundRepeat = "no-repeat";
    elem.style.backgroundPosition = "center";
    elem.style.color = color;

    if(text) {
        elem.innerHTML = text;
    }
}

function anim_loader(timestamp=3000) {
    setTimeout(()=> {
        createLoader('/loader2.gif');
        setTimeout(() => {
            getElement("#anim_loader").classList.add('rm-loader');
        }, timestamp)
    }, 10);
 }

function notification(parent, message, stop) {
    var el = addElement('div', parent, {
        "class":"text-center notif",
        "id":"model_notif_"
    });
    el.innerHTML = message;

    setTimeout(()=>{
        el.classList.add('slideIn');
        setTimeout(()=>{
            el.classList.remove('slideIn');
        }, stop);
    }, 1000);
}