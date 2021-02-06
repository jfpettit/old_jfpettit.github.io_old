---
layout: post
author: Jacob Pettit
comments: true
title: VSCode is dead! Long Live Vim!
featured-image: /assets/imgs/VS.gif
---

**Well, really Long Live NeoVim, but that just isn't as catchy is it?**

{: class="table-of-content"}
* TOC
{:toc}

A quick disclaimer to the emacs folks; I'm not trying to pick a fight here. I just like vim and never really got into using emacs, so I'm writing from my perspective.

# Some context

I've spent the last year plus using VSCode and generally had a great experience. I started off with default keybindings, until a coworker told me I should try vim keybindings.
After spending a little time learning to use the vim keys, I was editing code quicker and my mind was blown. :exploding_head:

This lasted until I started working on bigger projects and VSCode got kind of slow. I just suffered and kept using it until I saw a [YouTube video](https://youtu.be/gnupOrSEikQ) talking about how to configure vim like VSCode. I got inspired and had to try it for myself. Several hours and a couple of days worth of finetuning later, I had a setup that felt perfect. Now, I can't imagine going back anytime soon, and I like using my vim setup so much that I decided I had to write this blog post about it!

# [The long, long history of vim](https://en.wikipedia.org/wiki/Vim_(text_editor))

vim (Vi IMproved) was released in 1991 and started as a clone of the [vi text editor](https://en.wikipedia.org/wiki/Vi). Vi itself was originally written as the visual mode of a line-editor called [ex](https://en.wikipedia.org/wiki/Ex_(text_editor)) in 1976. 
ex was written by [Billy Joy](https://en.wikipedia.org/wiki/Bill_Joy) and Chuck Haley, and 
version 1.1 was part of the first edition of the Berkeley Software Distribution (BSD) Unix, released in March of 1978. 
In version 2 of ex, released as part of the Second BSD in May of 1979, the editor was released under the name vi.

vim had another forerunner, called Stevie (ST Editor for Vi Enthusiasts), which was created by Tim Thompson in 1987. 

Finally, in 1988, [Bram Moolenaar](https://en.wikipedia.org/wiki/Bram_Moolenaar) started his work on vim. The first public release was in 1991.
For years, folks continued to use vim, and then NeoVim was released in 2014.

So, vim is old and we shouldn't care about it right? Wrong.

# I swear, vim is still relevant

[Stack Overflow's 2019 Developer Survey](https://insights.stackoverflow.com/survey/2019#development-environments-and-tools) shows that vim is still in the top 5 editors used by developers!

You may be wondering why a text editor born 30 years ago (as of 2021) is still in use today. Hasn't technology gotten better? And haven't text editors gotten better alongside the rest of technology?

While technology in general has certainly gotten better, my answer for text editors is: Kind of, but not really. 

I think what's mostly happened is that text editors have gotten good GUIs (Graphical User Interfaces), and many people gravitate towards those. 
It makes sense; after all, it does feel more natural to use your mouse to navigate around a computer, especially when you're starting out.

However, all that convenience does come at a price. It is vastly less efficient to rely on a mouse for pointing, selecting, scrolling, and navigating around files than it is to use the keyboard to do these things. 
Vim sidesteps these problems, since it relies on keybindings to do all of this. 
Plus, vim is [widely](https://blog.onebar.io/hjkl-or-how-to-feel-less-tired-after-a-day-of-coding-48f975ba4091) [considered](https://news.ycombinator.com/item?id=7828717) [one](https://www.ibrahimirfan.com/why-you-should-learn-vim/) of the most ergonomically efficient editors. 
When you take the time to really learn vim, you can move around files at what feels like the speed of thought. And your hands and wrists will thank you after spending hours coding every day.

# Who should learn vim?

Anyone who is starting to feel slowed down by their text editor of choice should learn vim. 
Often, it can be really easy to pick up vim since many popular text editors have a vim plugin. 
These plugins let you use vim commands and keybindings without leaving the editor you're used to. 
vim is also very configurable, and while this is nice, it can also be intimidating when first starting out. Using a plugin in an editor you're alreadyused to helps soften the learning curve for using vim. As a related sidenote; if you want a nice introduction to learning vim commands, check out this [online game](https://vim-adventures.com).

If you're new to programming, like maybe you're in your first CS class or you've just started your first project, you probably don't need to learn vim. Programming already has quite a learning curve; you don't need to compound it by learning vim at the same time.

On the other side, if you're comfortable programming and you've done at least a few CS classes or you've got a couple of projects under your belt, you'll likely benefit from vim. 
When you first start learning vim, you'll move slower and might be a little less productive, but once you've put in the time to get familiar with it, your productivity will increase.

# That's all cool, but I still love VSCode...

And don't get me wrong, VSCode is great. It was the number one text editor on [Stack Overflow's 2019 Developer Survey](https://insights.stackoverflow.com/survey/2019#development-environments-and-tools). 
I mentioned above that I used VSCode for a bit before turning to vim, and I've gotta say, I've been able to replicate every feature I cared about from VSCode and it's all quicker in vim. 
Part of my reason for switching was that I'd check the activity manager on my machine and it would show that VSCode was sometimes using half a gigabyte of RAM! That felt a little silly to me, since all I was doing was editing text. Of course, part of that RAM usage is from language servers, but it still felt like too much. I think there are also things that VSCode does that I just never used. With vim, I can configure exactly what features I'm going to use and there's much less bloat, since I had to install those features directly.

If you're a VSCode user and you're happy, then great! I'm only kind of trying to convince you to use vim. But if you're a VSCode user and you wish you could keep only certain parts of the VSCode experience because the whole thing is too much for you... Then you should definitely check out vim.

# What's the point?

The point was for me to talk about how much I love vim and how I like my new setup, and to try to convince you to try it out too.

While you're here, I'll share my `init.vim` file with you. 
Credit to Ben Awad for providing a helpful [starter `init.vim`](https://gist.github.com/benawad/b768f5a5bbd92c8baabd363b7e79786f).
I started my file from his and added/modified what I wanted to change.

So here it is:

```vim
set nocompatible
set number
set autoindent
set smarttab
set shortmess+=I
set relativenumber
set laststatus=2
set backspace=indent,eol,start
set hidden
set ignorecase
set smartcase
set incsearch
nmap Q <Nop> " 'Q' in normal mode enters Ex mode. You almost never want this.
set noerrorbells visualbell t_vb=
set mouse+=a

" Below enables fuzzy file finding. In NORMAL mode, type :find <file>. Can
" autocomplete with tab.
set path+=**
set wildmenu
let mapleader = ","
" Keep cursor in middle of screen.
set so=999

" Setup plugin manager
call plug#begin('~/.vim/plugged')

" Main leanguage server, should configure similar to VScode
Plug 'neoclide/coc.nvim', {'branch': 'release'} 

Plug 'preservim/nerdtree' " Filetree viewer

" View git status in nerdtree, shows stars by dirty files.
Plug 'Xuyuanp/nerdtree-git-plugin' 

Plug 'ryanoasis/vim-devicons' " Cool icons for each filetype

" Navigate between open panes using Ctrl + h/j/k/l
Plug 'christoomey/vim-tmux-navigator' 

Plug 'airblade/vim-gitgutter' " View git changes in the file gutter.
Plug 'preservim/nerdcommenter' " Extra commenting powers.

" Syntax higlighting support for a truly ridiculous number of languages.
Plug 'sheerun/vim-polyglot' 

Plug 'joshdick/onedark.vim' " Onedark theme.
Plug 'kaicataldo/material.vim', { 'branch': 'main' } " Material theme
Plug 'vim-airline/vim-airline' " Airline bar on the bottom
Plug 'jiangmiao/auto-pairs' " Autocomplete parentheses, braces, etc.
Plug 'psliwka/vim-smoothie' " Smoother scrolling in vim

" Preview markdown files in the internet browser.
Plug 'iamcco/markdown-preview.nvim', { 'do': { -> mkdp#util#install() }, 'for': ['markdown', 'vim-plug']} 

call plug#end()

" NERDTree configurations.
autocmd VimEnter * NERDTree | wincmd p
autocmd FileType nerdtree setlocal nolist
autocmd BufEnter * if tabpagenr('$') == 1 && winnr('$') == 1 && exists('b:NERDTree') && b:NERDTree.isTabTree() |
    \ quit | endif
autocmd BufWinEnter * silent NERDTreeMirror

" Remap commands.
nnoremap <leader>n :NERDTreeFocus<CR>
" Ctrl-n toggles nerdtree
nmap <C-n> :NERDTreeToggle<CR>
" Ctrl-m runs Markdown preview. Will only work in markdown files
nmap <C-m> <Plug>MarkdownPreview
" Ctrl-s stops Markdown preview. Only does stuff if you're already running a
" markdown preview.
nmap <C-s> <Plug>MarkdownPreviewStop

vmap ++ <plug>NERDCommenterToggle
nmap ++ <plug>NERDCommenterToggle

let g:NERDTreeGitStatusWithFlags=1

" Set color theme styles.
let g:material_theme_style = 'darker'
colorscheme material 

if (has('termguicolors'))
  set termguicolors
endif


let g:mkdp_auto_start = 0

" Remap tab to code autocompletion.
inoremap <silent><expr> <TAB>
		\ pumvisible() ? "\<C-n>" :
		\ <SID>check_back_space() ? "<TAB>" :
		\ coc#refresh()
inoremap <expr><S-TAB> pumvisible() ? "\<C-p>" : "\<C-h>"

function! s:check_back_space() abort
  let col = col('.') - 1
  return !col || getline('.')[col - 1]  =~# '\s'
endfunction

" Use <c-space> to trigger completion.
if has('nvim')
  inoremap <silent><expr> <c-space> coc#refresh()
else
  inoremap <silent><expr> <c-@> coc#refresh()
endif
```

*In theory* this should let you get the exact setup that I have, but of course often these sorts of things require a little trial-and-error to get working perfectly.

Well, I hope you enjoyed my ramblings about vim, dear reader. Hopefully you think vim is worth checking out! 

Thanks for reading. :v:

# References

- For the history of vim: 
	- https://en.wikipedia.org/wiki/Vim_(text_editor)
	- https://en.wikipedia.org/wiki/Vi
	- https://en.wikipedia.org/wiki/Ex_(text_editor)
	- https://en.wikipedia.org/wiki/Bill_Joy
	- https://en.wikipedia.org/wiki/Bram_Moolenaar

- For vim's relevance:
	- https://insights.stackoverflow.com/survey/2019#development-environments-and-tools 
	- https://blog.onebar.io/hjkl-or-how-to-feel-less-tired-after-a-day-of-coding-48f975ba4091
	- https://www.ibrahimirfan.com/why-you-should-learn-vim/
	- https://news.ycombinator.com/item?id=7828717

- For my setup:
	- Neovim: https://neovim.io
	- Python jump to definition/reference/etc: https://github.com/portante/pycscope
	- Status bar: https://github.com/vim-airline/vim-airline
	- Material theme: https://github.com/kaicataldo/material.vim
	- Markdown preview: https://github.com/iamcco/markdown-preview.nvim
	- Nerdcommenter: https://github.com/preservim/nerdcommenter
	- Python language server: https://github.com/fannheyward/coc-pyright
	- Nerdtree: https://github.com/preservim/nerdtree
	- Plugin manager: https://github.com/junegunn/vim-plug
	- Coc code autocompletion package: https://github.com/neoclide/coc.nvim
	- Nerdtree git plugin: https://github.com/Xuyuanp/nerdtree-git-plugin
	- Vim devicons: https://github.com/ryanoasis/vim-devicons
	- Vim Tmux navigator: https://github.com/christoomey/vim-tmux-navigator
	- Vim gitgutter: https://github.com/airblade/vim-gitgutter
	- Vim polyglot: https://github.com/sheerun/vim-polyglot
	- Vim smoothie: https://github.com/psliwka/vim-smoothie
	- Material theme: https://github.com/kaicataldo/material.vim
	- Vim airline: https://github.com/vim-airline/vim-airline
	- Auto pairs: https://github.com/jiangmiao/auto-pairs

- Ben Awad's stuff:
	- Video: https://youtu.be/gnupOrSEikQ
	- `init.vim`: https://gist.github.com/benawad/b768f5a5bbd92c8baabd363b7e79786f
