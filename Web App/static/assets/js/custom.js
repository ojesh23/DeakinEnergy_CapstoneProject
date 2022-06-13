/*--------------------------
	baguetteBoxOne map
---------------------------- */	
window.onload = function() {
    if(typeof oldIE === 'undefined' && Object.keys)
        hljs.initHighlighting();

    baguetteBox.run('.baguetteBoxOne');
    baguetteBox.run('.baguetteBoxTwo');
    baguetteBox.run('.baguetteBoxThree', {
        animation: 'fadeIn'
    });
    baguetteBox.run('.baguetteBoxFour', {
        buttons: false
    });
    baguetteBox.run('.baguetteBoxFive', {
        captions: function(element) {
            return element.getElementsByTagName('img')[0].alt;
        }
    });
};



(window.jQuery),

function($) {

    var Page = function() {
        this.$topNavbar = $("#navbar-menu"),
        this.$stickyElem = $("#sticky-nav"),
        this.$backToTop = $('#back-to-top'),
        this.$contactForm = $("#contact-form")
    };
    //
    Page.prototype.init = function () {
        var $this = this;
        //window related event
 
       

        //Handling the scroll event
        $(window).scroll(function(){
            if ($(this).scrollTop() > 100) {
                $this.$backToTop.fadeIn();
            } else {
                $this.$backToTop.fadeOut();
            }
        }); 

        //on click on navbar - Smooth Scroll To Anchor (requires jQuery Easing plugin)
        this.$topNavbar.on('click', function(event) {
            var $anchor = $(event.target);
            if ($($anchor.attr('href')).length > 0 && $anchor.is('a.nav-link')) {
                $('html, body').stop().animate({
                    scrollTop: $($anchor.attr('href')).offset().top - 0
                }, 1500, 'easeInOutExpo');
                event.preventDefault();    
            }
        });

        //back-to-top button
        $this.$backToTop.click(function(){
            $("html, body").animate({ scrollTop: 0 }, 1000);
            return false;
        });


        //init contact app if contact form added in a page
        if(this.$contactForm.length>0)
            $.ContactForm.init();

    },
    //init
    $.Page = new Page, $.Page.Constructor = Page
}(window.jQuery),

//initializing
function($) {
    "use strict";
    $.Page.init()
}(window.jQuery);


/* ---------------------------------------------- /*
	 * Preloader
	/* ---------------------------------------------- */


    "use strict"; // Start of use strict
    // jQuery for page scrolling feature - requires jQuery Easing plugin
    $('a.page-scroll').bind('click', function(event) {
        var $anchor = $(this);
        $('html, body').stop().animate({
            scrollTop: ($($anchor.attr('href')).offset().top - 50)
        }, 1250, 'easeInOutExpo');
        event.preventDefault();
    });
    // Highlight the top nav as scrolling occurs
    $('body').scrollspy({
        target: '.navbar-fixed-top',
        offset: 51
    })
    // Offset for Main Navigation
    $('#mainNav').affix({
        offset: {
            top: 100
        }
    })

/*----------Modal for Classifier Explaination------------*/
$(document).ready(function(){
    //For SGD
    $('#linkView').click(function() {
        $('#modalWrapper').stop().fadeTo(500,1);
    })

    $('#modalWrapper').click(function(){
        $('#modalWrapper').fadeTo(500,0, function(){ $(this).hide(); });
    });

    $('#modalHolder').click(function(e) {
       e.stopPropagation();
    });

    $('#spClose').click(function(){
      $('#modalWrapper').click();
    });

    $('#left1').click(function() {
      $('#modalCarousel').carousel('prev');
    });

    $('#right1').click(function() {
      $('#modalCarousel').carousel('next');
    });

    //  For LR
    $('#linkView2').click(function() {
        $('#modalWrapper2').stop().fadeTo(500,1);
    })

    $('#modalWrapper2').click(function(){
        $('#modalWrapper2').fadeTo(500,0, function(){ $(this).hide(); });
    });

    $('#modalHolder2').click(function(e) {
       e.stopPropagation();
    });

    $('#spClose2').click(function(){
      $('#modalWrapper2').click();
    });

    $('#left2').click(function() {
      $('#modalCarousel2').carousel('prev');
    });

    $('#right2').click(function() {
      $('#modalCarousel2').carousel('next');
    });

    //  For RF
    $('#linkView3').click(function() {
        $('#modalWrapper3').stop().fadeTo(500,1);
    })

    $('#modalWrapper3').click(function(){
        $('#modalWrapper3').fadeTo(500,0, function(){ $(this).hide(); });
    });

    $('#modalHolder3').click(function(e) {
       e.stopPropagation();
    });

    $('#spClose3').click(function(){
      $('#modalWrapper3').click();
    });

    $('#left3').click(function() {
      $('#modalCarousel3').carousel('prev');
    });

    $('#right3').click(function() {
      $('#modalCarousel3').carousel('next');
    });

    //  For RNN
    $('#linkView4').click(function() {
        $('#modalWrapper4').stop().fadeTo(500,1);
    })

    $('#modalWrapper4').click(function(){
        $('#modalWrapper4').fadeTo(500,0, function(){ $(this).hide(); });
    });

    $('#modalHolder4').click(function(e) {
       e.stopPropagation();
    });

    $('#spClose4').click(function(){
      $('#modalWrapper4').click();
    });

    $('#left4').click(function() {
      $('#modalCarousel4').carousel('prev');
    });

    $('#right4').click(function() {
      $('#modalCarousel4').carousel('next');
    });

    //  For SVD
    $('#linkView4').click(function() {
        $('#modalWrapper4').stop().fadeTo(500,1);
    })

    $('#modalWrapper4').click(function(){
        $('#modalWrapper4').fadeTo(500,0, function(){ $(this).hide(); });
    });

    $('#modalHolder4').click(function(e) {
       e.stopPropagation();
    });

    $('#spClose4').click(function(){
      $('#modalWrapper4').click();
    });

    $('#left4').click(function() {
      $('#modalCarousel4').carousel('prev');
    });

    $('#right4').click(function() {
      $('#modalCarousel4').carousel('next');
    });

    //  For SVC
    $('#linkView5').click(function() {
        $('#modalWrapper5').stop().fadeTo(500,1);
    })

    $('#modalWrapper5').click(function(){
        $('#modalWrapper5').fadeTo(500,0, function(){ $(this).hide(); });
    });

    $('#modalHolder5').click(function(e) {
       e.stopPropagation();
    });

    $('#spClose5').click(function(){
      $('#modalWrapper5').click();
    });

    $('#left5').click(function() {
      $('#modalCarousel5').carousel('prev');
    });

    $('#right5').click(function() {
      $('#modalCarousel5').carousel('next');
    });

    //  For XGB
    $('#linkView6').click(function() {
        $('#modalWrapper6').stop().fadeTo(500,1);
    })

    $('#modalWrapper6').click(function(){
        $('#modalWrapper6').fadeTo(500,0, function(){ $(this).hide(); });
    });

    $('#modalHolder6').click(function(e) {
       e.stopPropagation();
    });

    $('#spClose6').click(function(){
      $('#modalWrapper6').click();
    });

    $('#left6').click(function() {
      $('#modalCarousel6').carousel('prev');
    });

    $('#right6').click(function() {
      $('#modalCarousel6').carousel('next');
    });

    //  For Voting
    $('#linkView7').click(function() {
        $('#modalWrapper7').stop().fadeTo(500,1);
    })

    $('#modalWrapper7').click(function(){
        $('#modalWrapper7').fadeTo(500,0, function(){ $(this).hide(); });
    });

    $('#modalHolder7').click(function(e) {
       e.stopPropagation();
    });

    $('#spClose7').click(function(){
      $('#modalWrapper7').click();
    });

    $('#left7').click(function() {
      $('#modalCarousel7').carousel('prev');
    });

    $('#right7').click(function() {
      $('#modalCarousel7').carousel('next');
    });

});
	


