const fullname = document.getElementById("fullname");
const mssv = document.getElementById("mssv");
const in_time = document.getElementById("in_time");
const out_time = document.getElementById("out_time");
const status_name = document.getElementById("status");
const date = document.getElementById("date").innerHTML;
setInterval(() =>{      
  const xhttp = new XMLHttpRequest();
  xhttp.open("POST", "Getcalendar", true);
  xhttp.setRequestHeader('Content-type', 'application/x-www-form-urlencoded');
  xhttp.onreadystatechange = function() {//Call a function when the state changes.
      if(xhttp.readyState == 4 && xhttp.status == 200) {
          if(xhttp.responseText!="false"){  
            const obj = JSON.parse(xhttp.responseText);
            if(obj==""){
              console.log("rỗng")
              fullname.innerHTML="Unknown";
              mssv.innerHTML="Unknown";
              in_time.innerHTML="Unknown";
              out_time.innerHTML="Unknown";
              status_name.style.background= "red";
              status_name.innerHTML="Trạng thái";
            }else{
              fullname.innerHTML = obj[obj.length-1]["fullname"];
              mssv.innerHTML=obj[obj.length-1]["mssv"]
              in_time.innerHTML = obj[obj.length-1]["in_time"];
              out_time.innerHTML = obj[obj.length-1]["out_time"];
              if(date==obj[obj.length-1]["in_date"]){
                status_name.style.background= "blue";
                status_name.innerHTML="Đã điểm danh";
              }
            }
            

            // console.log(obj.length)
            // console.log(obj[obj.length-1]["fullname"])
            // console.log(xhttp.responseText)
          }
      }
  }
  xhttp.send();
},1000);