function handleMarqueScroll() {
    window.addEventListener("wheel", function(dets) {
        if (dets.deltaY > 0) {
            gsap.to(".marque", {
                transform: 'translateX(-200%)',
                duration: 5,
                ease: "none",
                repeat: -1
            });

            gsap.to(".marque i", {
                rotate: 180,
            });
        } else if (dets.deltaY < 0) {
            gsap.to(".marque", {
                transform: 'translateX(0%)',
                duration: 5,
                ease: "none",
                repeat: -1
            });

            gsap.to(".marque i", {
                rotate: 0,
            });
        }
    });
}

handleMarqueScroll();


function openPopup(projectId) {
    document.getElementById(projectId).style.display = 'block';
    console.log("clicked");
}

function closePopup(projectId) {
    document.getElementById(projectId).style.display = 'none';
    // Close all open details
    document.querySelectorAll('.popup__description_detail_opened').forEach(function(openDetail) {
        openDetail.classList.remove('popup__description_detail_opened');
        openDetail.previousElementSibling.querySelector('.popup__arrow').classList.remove('popup__arrow_opened');
    });
}