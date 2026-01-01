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
      
      # Parse type/style arguments
      types = %w(success info warning error example exploit)
      active_types = types.select { |t| @markup.include?(t) }
      
      title = nil
      if @markup =~ /title=(["'])(.*?)\1/
        title = $2
      end

      # Construct HTML
      if fold
        summary_attr = title ? " data-title=\"#{title}\"" : ""
        classes = [@block_name] + active_types
        "<details class=\"#{classes.join(' ')}\" markdown=\"1\">\n<summary#{summary_attr}></summary>\n#{text}\n</details>"
      else
        classes = [@block_name] + active_types
        classes << "inline" if inline
        class_attr = " class=\"#{classes.join(' ')}\""
        
        title_attr = title ? " data-title=\"#{title}\"" : ""
        
        "<div#{class_attr}#{title_attr} markdown=\"1\">\n#{text}\n</div>"
      end
    end
  end
end

# Register tags
%w(proof theorem lemma proposition note remark example solution code_block).each do |tag|
  Liquid::Template.register_tag(tag, Jekyll::CustomBlock)
end
