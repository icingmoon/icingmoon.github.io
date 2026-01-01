module Jekyll
  class CustomBlock < Liquid::Block
    def initialize(tag_name, markup, tokens)
      super
      @block_name = tag_name
      @markup = markup
    end

    def render(context)
      text = super
      
      # Parse arguments
      fold = @markup.include?('fold')
      inline = @markup.include?('inline')
      
      title = nil
      if @markup =~ /title=(["'])(.*?)\1/
        title = $2
      end

      # Construct HTML
      if fold
        summary_attr = title ? " data-title=\"#{title}\"" : ""
        "<details class=\"#{@block_name}\" markdown=\"1\">\n<summary#{summary_attr}></summary>\n#{text}\n</details>"
      else
        classes = [@block_name]
        classes << "inline" if inline
        class_attr = " class=\"#{classes.join(' ')}\""
        
        title_attr = title ? " data-title=\"#{title}\"" : ""
        
        "<div#{class_attr}#{title_attr} markdown=\"1\">\n#{text}\n</div>"
      end
    end
  end
end

# Register tags
%w(proof theorem lemma proposition note remark).each do |tag|
  Liquid::Template.register_tag(tag, Jekyll::CustomBlock)
end
