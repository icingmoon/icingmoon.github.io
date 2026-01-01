(function() {
  var lang = document.documentElement.lang || 'en';
  var isZh = lang.startsWith('zh');
  
  var i18n = {
    copy: isZh ? '复制' : 'Copy',
    copied: isZh ? '已复制!' : 'Copied!',
    error: isZh ? '错误' : 'Error'
  };

  function createCopyButton(codeBlock) {
    var button = document.createElement('button');
    button.className = 'copy-code-button';
    button.type = 'button';
    button.innerText = i18n.copy;
    
    button.addEventListener('click', function() {
      var code = codeBlock.querySelector('code').innerText;
      navigator.clipboard.writeText(code).then(function() {
        button.innerText = i18n.copied;
        button.classList.add('copied');
        setTimeout(function() {
          button.innerText = i18n.copy;
          button.classList.remove('copied');
        }, 2000);
      }, function(err) {
        console.error('Could not copy text: ', err);
        button.innerText = i18n.error;
      });
    });
    
    return button;
  }

  // Find all code blocks
  // Jekyll usually wraps code in div.highlighter-rouge
  var codeBlocks = document.querySelectorAll('div.highlighter-rouge');
  
  codeBlocks.forEach(function(wrapper) {
    // Check if it's inside a custom block that might already handle positioning
    // But generally we want the button inside the wrapper
    wrapper.style.position = 'relative';
    
    var button = createCopyButton(wrapper);
    wrapper.appendChild(button);
  });
})();
