(function() {
  // Find all code blocks that have attributes but weren't processed by the Ruby plugin
  // This happens when plugins are disabled (e.g. GitHub Pages Safe Mode)
  var codeBlocks = document.querySelectorAll('div.highlighter-rouge[title], div.highlighter-rouge[fold], div.highlighter-rouge[type]');
  
  codeBlocks.forEach(function(block) {
    var title = block.getAttribute('title');
    var fold = block.getAttribute('fold');
    var type = block.getAttribute('type');
    
    // Create wrapper
    var wrapper;
    if (fold) {
      wrapper = document.createElement('details');
      if (fold === 'open') {
        wrapper.open = true;
      }
      
      var summary = document.createElement('summary');
      if (title) {
        summary.setAttribute('data-title', title);
      }
      wrapper.appendChild(summary);
    } else {
      wrapper = document.createElement('div');
      if (title) {
        wrapper.setAttribute('data-title', title);
      }
    }
    
    // Add classes
    wrapper.classList.add('code_block');
    if (type) {
      wrapper.classList.add(type);
    }
    
    // Insert wrapper before block
    block.parentNode.insertBefore(wrapper, block);
    
    // Move block into wrapper
    wrapper.appendChild(block);
    
    // Clean up attributes on the original block to avoid confusion
    block.removeAttribute('title');
    block.removeAttribute('fold');
    block.removeAttribute('type');
  });
})();
